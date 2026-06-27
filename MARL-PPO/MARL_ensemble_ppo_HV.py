import os
import time
import warnings
import joblib

# PENGATURAN CRITICAL: Cegah CPU Hang & Error Iterator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from pymoo.indicators.hv import HV

# Sembunyikan Warning agar terminal bersih
warnings.filterwarnings("ignore")

# =====================
# === CONFIG ==========
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, "best_ann_surrogate_model_LH2_FINAL.keras")
SCALER_X_PATH = os.path.join(SCRIPT_DIR, "x_scaler.gz")
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, "y_scaler.gz")

INPUT_COLS = ['Pout_K21 (kPa)', 'Pout_VLV101 (kPa)', 'Pout_X121 (kPa)']
OBJECTIVE_COLS = ['SEC (kWh/kg)', 'Efisiensi Exergy (%)', 'LCOH (USD/kg)']
LOW_BOUND = np.array([300., 110., 120.])
HIGH_BOUND = np.array([700., 150., 150.])

# Hyperparameters untuk Batch 30 Runs
N_AGENTS = 3 
EPISODES_PER_AGENT = 800 
MAX_STEPS_PER_EPISODE = 8
LR = 3e-4
K_EPOCHS = 5 
DEVICE = torch.device("cpu")

# =====================
# === LOAD SURROGATE ==
# =====================
print("--- Menginisialisasi Model ANN ---")
try:
    # Load model tanpa compile untuk kecepatan
    SURROGATE_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Bungkus model dengan tf.function untuk kecepatan eksekusi 10x lebih cepat
    @tf.function(reduce_retracing=True)
    def model_predict(x):
        return SURROGATE_MODEL(x, training=False)

    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y = joblib.load(SCALER_Y_PATH)
    print("✅ Model Surrogate & Scaler Siap.")
except Exception as e:
    print(f"❌ ERROR LOAD FILE: {e}")
    exit()

# =====================
# === ENVIRONMENT =====
# =====================
class SurrogateEnv:
    def __init__(self):
        self.max_steps = MAX_STEPS_PER_EPISODE
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return np.random.rand(3).astype(np.float32)

    def step(self, action: np.ndarray):
        self.current_step += 1
        a = np.clip(action, 0.0, 1.0)
        real_input = LOW_BOUND + a * (HIGH_BOUND - LOW_BOUND)
        
        # Transformasi input dengan DataFrame agar StandardScaler senang
        df_in = pd.DataFrame(real_input.reshape(1, -1), columns=INPUT_COLS)
        scaled = SCALER_X.transform(df_in).astype(np.float32)
        
        # REVISI FINAL: Memanggil model secara langsung (Fungsional)
        # Menghindari error "make_iterator" dan jauh lebih cepat
        pred_scaled = model_predict(scaled).numpy()
        pred = SCALER_Y.inverse_transform(pred_scaled)[0]

        # Reward: [SEC(min), Exergy(max), LCOH(min)]
        reward_vec = np.array([-pred[0], pred[1], -pred[2]], dtype=np.float32)
        done = self.current_step >= self.max_steps
        next_state = np.clip(a + 0.01 * (np.random.rand(3) - 0.5), 0.0, 1.0)
        
        return next_state.astype(np.float32), reward_vec, done, {"real_input": real_input, "out": pred}

# =====================
# === NETWORKS ========
# =====================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3), nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros(3))
    def forward(self, s): return self.net(s)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, s): return self.net(s).squeeze(-1)

# =====================
# === AGENT & TRAIN ===
# =====================
def train_one_run(run_idx):
    env = SurrogateEnv()
    actors = [Actor().to(DEVICE) for _ in range(N_AGENTS)]
    critic = Critic().to(DEVICE)
    optimizer = optim.Adam(list(critic.parameters()) + [p for a in actors for p in a.parameters()], lr=LR)
    
    weights = [np.eye(3)[i] if i < 3 else np.random.dirichlet(np.ones(3)) for i in range(N_AGENTS)]
    all_pareto_outputs = []

    for a_idx, w in enumerate(weights):
        for ep in range(1, EPISODES_PER_AGENT + 1):
            state, done = env.reset(), False
            states, actions, logprobs, rewards = [], [], [], []
            
            while not done:
                s_t = torch.FloatTensor(state).to(DEVICE)
                mu = actors[a_idx](s_t)
                std = torch.exp(torch.clamp(actors[a_idx].log_std, -2.0, 0.5))
                dist = Normal(mu, std)
                act_raw = dist.sample()
                lp = dist.log_prob(act_raw).sum()
                act_m = ((act_raw.detach().cpu().numpy() + 1.0) / 2.0).astype(np.float32)
                
                next_s, rew_v, done, info = env.step(act_m)
                
                states.append(s_t); actions.append(act_raw); logprobs.append(lp); rewards.append(rew_v)
                state = next_s
                if done: all_pareto_outputs.append(info['out'])

            # Update PPO
            optimizer.zero_grad()
            s_b, a_b, lp_b = torch.stack(states), torch.stack(actions), torch.stack(logprobs)
            r_v = np.vstack(rewards)
            scal_rew = ((r_v - r_v.mean(0)) / (r_v.std(0) + 1e-8)).dot(w)
            
            vals = critic(s_b).detach().cpu().numpy()
            ret = np.zeros_like(scal_rew)
            for t in reversed(range(len(scal_rew))):
                ret[t] = scal_rew[t] + 0.99 * (ret[t+1] if t+1 < len(scal_rew) else 0)
            
            ret_t = torch.FloatTensor(ret).to(DEVICE)
            for _ in range(K_EPOCHS):
                curr_mu = actors[a_idx](s_b)
                curr_dist = Normal(curr_mu, torch.exp(torch.clamp(actors[a_idx].log_std, -2.0, 0.5)))
                curr_lp = curr_dist.log_prob(a_b).sum(-1)
                curr_vals = critic(s_b)
                
                ratio = torch.exp(curr_lp - lp_b.detach())
                surr1 = ratio * (ret_t - curr_vals.detach())
                surr2 = torch.clamp(ratio, 0.8, 1.2) * (ret_t - curr_vals.detach())
                
                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(curr_vals, ret_t)
                loss.backward()
                optimizer.step()
        
        print(f"      > Run {run_idx} | Agent {a_idx+1} Selesai...", flush=True)

    return pd.DataFrame(all_pareto_outputs, columns=OBJECTIVE_COLS)

# =====================
# === MAIN BATCH ======
# =====================
if __name__ == "__main__":
    n_runs = 30
    hv_results = []
    # Ref Point: SEC=5, Exergy=-60 (minimasi), LCOH=3.5
    hv_calculator = HV(ref_point=np.array([5.0, -60.0, 3.5]))
    
    print(f"🚀 Memulai Batch Optimization MORL (30 Runs)")
    print(f"Estimasi Waktu: ~1 Jam. Hasil akan otomatis tersimpan di folder script.")
    
    start_time = time.time()
    for i in range(1, n_runs + 1):
        run_start = time.time()
        print(f"\n[RUN {i}/{n_runs}] Sedang Berjalan...", flush=True)
        
        np.random.seed(i + 100); torch.manual_seed(i + 100)
        
        # Jalankan 1 Run
        p_out = train_one_run(i)
        
        # Hitung HV
        pts_hv = p_out.values.copy()
        pts_hv[:, 1] = -pts_hv[:, 1] # Balik Exergy agar menjadi minimasi
        
        try:
            val_hv = hv_calculator(pts_hv)
            hv_results.append(val_hv)
            print(f"   ✅ Run {i} Selesai | HV: {val_hv:.6f} | Waktu: {time.time()-run_start:.1f}s")
        except Exception as e:
            hv_results.append(0)
            print(f"   ⚠️ Run {i} Selesai | HV Gagal dihitung: {e}")

    # Simpan ke CSV
    path_final = os.path.join(SCRIPT_DIR, "hv_morl_30_runs.csv")
    pd.DataFrame({"Run": range(1, 31), "HV_MORL": hv_results}).to_csv(path_final, index=False)
    
    print(f"\n🎉 SEMUA PROSES SELESAI!")
    print(f"File Hasil: {path_final}")
    print(f"Total Waktu: {(time.time()-start_time)/60:.1f} Menit.")