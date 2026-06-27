# morl_ensemble_ppo.py
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.cuda.amp import autocast, GradScaler

# =====================
# === CONFIG ==========
# =====================
# paths (ubah bila perlu)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_ann_surrogate_model_LH2_FINAL.keras")
SCALER_X_PATH = os.path.join(SCRIPT_DIR, "x_scaler.gz")
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, "y_scaler.gz")

# problem definition
INPUT_COLS = ['Pout_K21 (kPa)', 'Pout_VLV101 (kPa)', 'Pout_X121 (kPa)']
OBJECTIVE_COLS = ['SEC (kWh/kg)', 'Efisiensi Exergy (%)', 'LCOH (USD/kg)']
INPUT_DIM = len(INPUT_COLS)
OUTPUT_DIM = len(OBJECTIVE_COLS)

# bounds (real_input = low + action*(high-low))
LOW_BOUND = np.array([300., 110., 120.])
HIGH_BOUND = np.array([700., 150., 150.])

# ensemble & training hyperparams
N_AGENTS = 3                     
NUM_WEIGHT_COMBINATIONS = N_AGENTS
EPISODES_PER_AGENT = 800         
MAX_STEPS_PER_EPISODE = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
EPS_CLIP = 0.2
K_EPOCHS = 8
ENTROPY_COEF = 0.03
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5
GRAD_CLIP = 0.5
BATCH_UPDATES_PER_EP = 1         

# misc
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
MIXED_PRECISION = USE_CUDA       

SAVE_PREFIX = "morl_ensemble"
SEED = 42

# set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

# =====================
# === LOAD SURROGATE ==
# =====================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("Loading surrogate model and scalers...")
try:
    SURROGATE_MODEL = tf.keras.models.load_model(MODEL_PATH)
    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y = joblib.load(SCALER_Y_PATH)
    print("✅ Loaded surrogate and scalers.")
except Exception as e:
    raise IOError(f"GAGAL MEMUAT FILE PENDUKUNG. Pastikan file .keras dan .gz ada di folder yang sama.\nError: {e}")

# =====================
# === ENVIRONMENT =====
# =====================
class SurrogateEnv:
    def __init__(self, surrogate_model, scaler_x, scaler_y, low_bound, high_bound, max_steps=8):
        self.surrogate_model = surrogate_model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.low = low_bound
        self.high = high_bound
        self.input_dim = len(low_bound)
        self.max_steps = max_steps
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        self.state = np.random.rand(self.input_dim).astype(np.float32)
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        self.current_step += 1
        a = np.clip(action, 0.0, 1.0)
        real_input = self.low + a * (self.high - self.low)
        scaled = self.scaler_x.transform(real_input.reshape(1, -1))
        pred_scaled = self.surrogate_model.predict(scaled, verbose=0)
        pred = self.scaler_y.inverse_transform(pred_scaled)[0]

        # reward vector: convert to "higher is better" 
        # pred[0] = SEC (Min) -> -pred[0]
        # pred[1] = Exergy (Max) -> +pred[1]
        # pred[2] = LCOH (Min) -> -pred[2]
        reward_vec = np.array([-pred[0], pred[1], -pred[2]], dtype=np.float32)

        done = self.current_step >= self.max_steps
        next_state = a + 0.01 * (np.random.rand(self.input_dim).astype(np.float32) - 0.5)
        next_state = np.clip(next_state, 0.0, 1.0)
        info = {"real_input": real_input, "predicted_output": pred}
        self.state = next_state.copy()
        return next_state, reward_vec, done, info

# =====================
# === NETWORKS ========
# =====================
class SharedCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()   # outputs in [-1,1]
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        return self.net(s)

# =====================
# === AGENT WRAPPER ===
# =====================
class EnsembleAgent:
    def __init__(self, n_agents: int, state_dim: int, action_dim: int):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.actors = [Actor(state_dim, action_dim).to(DEVICE) for _ in range(n_agents)]
        self.critic = SharedCritic(state_dim).to(DEVICE)
        
        params = []
        for a in self.actors:
            params += list(a.parameters())
        params += list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=LR)
        self.grad_scaler = GradScaler() if MIXED_PRECISION else None

        for actor in self.actors:
            with torch.no_grad():
                actor.log_std.uniform_(LOG_STD_MIN, LOG_STD_MAX)

    def select_action(self, agent_idx: int, state_np: np.ndarray):
        actor = self.actors[agent_idx]
        state = torch.FloatTensor(state_np).to(DEVICE)
        with torch.no_grad():
            mu = actor(state)   # [-1,1]
            log_std = torch.clamp(actor.log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            sampled = dist.sample()
            logprob = dist.log_prob(sampled).sum()
            action_mapped = ((sampled.cpu().numpy() + 1.0) / 2.0).astype(np.float32)
        return action_mapped, sampled.cpu(), logprob.cpu()

    def update(self, memories: dict, weights: np.ndarray):
        device = DEVICE
        all_agent_losses = 0.0
        
        states_list = []
        actions_list = []
        old_logprobs_list = []
        returns_list = []
        advantages_list = []

        for a_idx in range(self.n_agents):
            mem = memories[a_idx]
            if len(mem['states']) == 0:
                continue
            
            states = torch.stack(mem['states']).to(device)          
            actions = torch.stack(mem['actions']).to(device)        
            old_logprobs = torch.stack(mem['logprobs']).to(device)  
            rewards_vec = np.vstack(mem['rewards'])                 

            eps = 1e-8
            rv_norm = (rewards_vec - rewards_vec.mean(axis=0)) / (rewards_vec.std(axis=0) + eps)
            scalar_rewards = rv_norm.dot(weights)  

            with torch.no_grad():
                vals = self.critic(states).detach().cpu().numpy()   
            
            T = len(scalar_rewards)
            returns = np.zeros(T, dtype=np.float32)
            advantages = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            next_value = 0.0
            for t in reversed(range(T)):
                delta = scalar_rewards[t] + GAMMA * next_value - vals[t]
                last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
                advantages[t] = last_gae
                returns[t] = advantages[t] + vals[t]
                next_value = vals[t]
            
            # <--- REVISI: Menggunakan torch.float32
            returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
            advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            states_list.append(states)
            actions_list.append(actions)
            old_logprobs_list.append(old_logprobs)
            returns_list.append(returns_t)
            advantages_list.append(advantages_t)

        if len(states_list) == 0:
            return

        states_b = torch.cat(states_list, dim=0)             
        actions_b = torch.cat(actions_list, dim=0)           
        old_logprobs_b = torch.cat(old_logprobs_list, dim=0) 
        returns_b = torch.cat(returns_list, dim=0)           
        advantages_b = torch.cat(advantages_list, dim=0)     

        for epoch in range(K_EPOCHS):
            start = 0
            total_loss = 0.0
            self.optimizer.zero_grad()
            for a_idx in range(self.n_agents):
                mem = memories[a_idx]
                T = len(mem['states'])
                if T == 0:
                    continue
                end = start + T
                states_a = states_b[start:end]
                actions_a = actions_b[start:end]
                
                # <--- REVISI CRITICAL: Menambahkan .view(-1) agar shape sesuai
                old_logprobs_a = old_logprobs_b[start:end].view(-1)
                
                returns_a = returns_b[start:end]
                adv_a = advantages_b[start:end]

                actor = self.actors[a_idx]
                
                # Forward Pass
                if MIXED_PRECISION:
                    with autocast():
                        mu = actor(states_a)
                        log_std = torch.clamp(actor.log_std, LOG_STD_MIN, LOG_STD_MAX)
                        std = torch.exp(log_std)
                        dist = Normal(mu, std)
                        logprobs = dist.log_prob(actions_a).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()
                        values = self.critic(states_a)
                else:
                    mu = actor(states_a)
                    log_std = torch.clamp(actor.log_std, LOG_STD_MIN, LOG_STD_MAX)
                    std = torch.exp(log_std)
                    dist = Normal(mu, std)
                    logprobs = dist.log_prob(actions_a).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    values = self.critic(states_a)

                # Ratio calculation (sekarang dimensinya aman)
                ratios = torch.exp(logprobs - old_logprobs_a)
                surr1 = ratios * adv_a
                surr2 = torch.clamp(ratios, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv_a
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = nn.MSELoss()(values, returns_a)
                
                loss = loss_actor + 0.5 * loss_critic - ENTROPY_COEF * entropy
                total_loss = total_loss + loss
                start = end

            if MIXED_PRECISION:
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()) + [p for a in self.actors for p in a.parameters()], max_norm=GRAD_CLIP)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()) + [p for a in self.actors for p in a.parameters()], max_norm=GRAD_CLIP)
                self.optimizer.step()

# =====================
# === PARETO UTILS ====
# =====================
def compute_pareto(points: np.ndarray, minimization_mask: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    pts = points.copy()
    for j in range(pts.shape[1]):
        if minimization_mask[j]:
            pts[:, j] = -pts[:, j]
    N = pts.shape[0]
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        if not is_pareto[i]:
            continue
        better_or_equal = np.all(pts >= pts[i] - tol, axis=1)
        strictly_better = np.any(pts > pts[i] + tol, axis=1)
        dominates = better_or_equal & strictly_better
        if np.any(dominates):
            is_pareto[i] = False
    return is_pareto

# =====================
# === TRAINING LOOP ===
# =====================
def create_weight_list(n_agents: int) -> List[np.ndarray]:
    if n_agents == 1:
        return [np.array([1.0, 0.0, 0.0])]
    base = []
    for i in range(OUTPUT_DIM):
        w = np.zeros(OUTPUT_DIM)
        w[i] = 1.0
        base.append(w)
    if n_agents <= OUTPUT_DIM:
        return base[:n_agents]
    else:
        extras = [np.random.dirichlet(np.ones(OUTPUT_DIM)) for _ in range(n_agents - OUTPUT_DIM)]
        return base + extras

def train_ensemble(env: SurrogateEnv, n_agents: int = N_AGENTS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensemble = EnsembleAgent(n_agents=n_agents, state_dim=INPUT_DIM, action_dim=INPUT_DIM)
    weight_list = create_weight_list(n_agents)

    all_inputs = []
    all_outputs = []

    # Minimization mask: [SEC (min), Exergy (max -> false), LCOH (min)]
    minimization_mask = np.array([True, False, True]) 

    start_time = time.time()
    for a_idx, w in enumerate(weight_list):
        print(f"\n=== Agent {a_idx+1}/{n_agents} training (weight={np.round(w,3)}) ===")
        for ep in range(1, EPISODES_PER_AGENT + 1):
            state = env.reset()
            memories = {i: {'states': [], 'actions': [], 'logprobs': [], 'rewards': []} for i in range(n_agents)}
            done = False
            t = 0
            
            while not done and t < MAX_STEPS_PER_EPISODE:
                for agent_j in range(n_agents):
                    action_mapped, action_raw, logprob = ensemble.select_action(agent_j, state)
                    next_state, reward_vec, done, info = env.step(action_mapped)
                    
                    memories[agent_j]['states'].append(torch.FloatTensor(state))
                    memories[agent_j]['actions'].append(action_raw)
                    memories[agent_j]['logprobs'].append(torch.FloatTensor([logprob.item()]))
                    memories[agent_j]['rewards'].append(reward_vec.astype(np.float32))
                    
                    all_inputs.append(info['real_input'])
                    all_outputs.append(info['predicted_output'])
                    
                    state = next_state
                    t += 1
                    if done or t >= MAX_STEPS_PER_EPISODE:
                        break
            ensemble.update(memories, w)

            if ep % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Agent {a_idx+1} Ep {ep}/{EPISODES_PER_AGENT}  elapsed {elapsed:.1f}s  total samples {len(all_outputs)}")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.1f} seconds. Total samples: {len(all_outputs)}")

    all_out_df = pd.DataFrame(all_outputs, columns=OBJECTIVE_COLS)
    all_in_df = pd.DataFrame(all_inputs, columns=INPUT_COLS)

    is_p = compute_pareto(all_out_df.values, minimization_mask, tol=1e-4)
    pareto_in = all_in_df[is_p].reset_index(drop=True)
    pareto_out = all_out_df[is_p].reset_index(drop=True)

    return pareto_in, pareto_out, all_in_df, all_out_df

# =====================
# === PLOTTING =========
# =====================
def plot_pairwise_and_3d(pareto_df: pd.DataFrame, all_df: pd.DataFrame, save_dir, prefix):
    # Buat full path untuk prefix
    full_prefix = os.path.join(save_dir, prefix)
    
    n = len(OBJECTIVE_COLS)
    for i in range(n):
        for j in range(i+1, n):
            obj1, obj2 = OBJECTIVE_COLS[i], OBJECTIVE_COLS[j]
            plt.figure(figsize=(8,6))
            plt.scatter(all_df[obj1], all_df[obj2], c='gray', alpha=0.12, s=12, label='Explored')
            plt.scatter(pareto_df[obj1], pareto_df[obj2], c='red', s=50, edgecolors='k', label='Pareto')
            plt.xlabel(obj1); plt.ylabel(obj2)
            plt.title(f'{obj1} vs {obj2}')
            plt.grid(linestyle='--', alpha=0.5)
            plt.legend()
            
            # Simpan dengan path lengkap
            fn = f"{full_prefix}_2d_{i}_{j}.png"
            plt.tight_layout()
            plt.savefig(fn, dpi=300)
            print(f"Gambar disimpan: {fn}")
            # plt.show() # Opsional: dimatikan agar tidak memblokir proses jika ditinggal

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(all_df[OBJECTIVE_COLS[0]], all_df[OBJECTIVE_COLS[1]], all_df[OBJECTIVE_COLS[2]], alpha=0.08, s=8, label='Explored')
        ax.scatter(pareto_df[OBJECTIVE_COLS[0]], pareto_df[OBJECTIVE_COLS[1]], pareto_df[OBJECTIVE_COLS[2]], c='red', s=50, label='Pareto')
        ax.set_xlabel(OBJECTIVE_COLS[0]); ax.set_ylabel(OBJECTIVE_COLS[1]); ax.set_zlabel(OBJECTIVE_COLS[2])
        ax.set_title('Pareto Front 3D')
        plt.legend()
        
        fn3 = f"{full_prefix}_pareto_3d.png"
        plt.savefig(fn3, dpi=300)
        print(f"Gambar 3D disimpan: {fn3}")
        # plt.show() 
    except Exception as e:
        print("3D plotting failed:", e)

# =====================
# === MAIN ============
# =====================
if __name__ == "__main__":
    print(f"Device: {DEVICE}, Mixed precision: {MIXED_PRECISION}")
    
    # 1. Pastikan Path Penyimpanan Benar (Di folder yang sama dengan script)
    # SCRIPT_DIR sudah didefinisikan di atas, kita gunakan itu.
    print(f"Folder Penyimpanan Target: {SCRIPT_DIR}")
    
    env = SurrogateEnv(SURROGATE_MODEL, SCALER_X, SCALER_Y, LOW_BOUND, HIGH_BOUND, max_steps=MAX_STEPS_PER_EPISODE)

    # 2. Jalankan Training
    pareto_in, pareto_out, all_in, all_out = train_ensemble(env, n_agents=N_AGENTS)

    # 3. Simpan ke EXCEL (.xlsx) dengan Absolute Path
    print("\nMenyimpan data ke Excel...")
    
    # Gabungkan path folder dengan nama file
    path_all = os.path.join(SCRIPT_DIR, f"{SAVE_PREFIX}_all_results.xlsx")
    path_pareto = os.path.join(SCRIPT_DIR, f"{SAVE_PREFIX}_pareto_results.xlsx")
    
    # Simpan All Results
    all_results = pd.concat([all_in, all_out], axis=1)
    all_results.to_excel(path_all, index=False)
    
    # Simpan Pareto Results
    pareto_results = pd.concat([pareto_in, pareto_out], axis=1)
    pareto_results.to_excel(path_pareto, index=False)
    
    print(f"✅ SUKSES: Data tersimpan di:\n  1. {path_all}\n  2. {path_pareto}")

    # 4. Plotting
    print("Membuat grafik...")
    plot_pairwise_and_3d(pareto_out, all_out, save_dir=SCRIPT_DIR, prefix=SAVE_PREFIX)
    print("🎉 Selesai. Cek folder Anda sekarang.")