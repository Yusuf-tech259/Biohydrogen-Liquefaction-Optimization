import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import warnings
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV

# Matikan warning agar terminal bersih dan proses lebih cepat
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =====================
# === KONFIGURASI ===
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_ann_surrogate_model_LH2_FINAL.keras')
SCALER_X_PATH = os.path.join(SCRIPT_DIR, 'x_scaler.gz')
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, 'y_scaler.gz')

INPUT_COLS = ['Pout_K21 (kPa)', 'Pout_VLV101 (kPa)', 'Pout_X121 (kPa)']

# --- Memuat Surrogate Model dan Scaler ---
try:
    # Menggunakan pemanggilan fungsional untuk kecepatan (seperti pada MORL)
    SURROGATE_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    @tf.function(reduce_retracing=True)
    def model_predict(x):
        return SURROGATE_MODEL(x, training=False)

    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y = joblib.load(SCALER_Y_PATH)
    print("✅ Model dan scaler berhasil dimuat.")
except Exception as e:
    raise IOError(f"GAGAL MEMUAT FILE: {e}")

# =====================
# === PROBLEM DEF ===
# =====================
class SurrogateProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=3, n_obj=3, n_constr=0,
            xl=np.array([300, 110, 120]), 
            xu=np.array([700, 150, 150])   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Gunakan DataFrame untuk menghindari UserWarning
        df_in = pd.DataFrame(x, columns=INPUT_COLS)
        input_scaled = SCALER_X.transform(df_in).astype(np.float32)
        
        # Prediksi fungsional (lebih cepat)
        pred_scaled = model_predict(input_scaled).numpy()
        responses = SCALER_Y.inverse_transform(pred_scaled)
        
        # F: [SEC (min), -Exergy (max), LCOH (min)]
        out["F"] = np.column_stack([responses[:, 0], -responses[:, 1], responses[:, 2]])

# =====================
# === BATCH RUNNING ===
# =====================
if __name__ == '__main__':
    n_runs = 30
    hv_results_nsga = []
    
    # Reference Point: HARUS SAMA DENGAN MORL [SEC_max, -Exergy_min, LCOH_max]
    ref_point = np.array([5.0, -60.0, 3.5])
    hv_calculator = HV(ref_point=ref_point)

    problem = SurrogateProblem()

    print(f"\n🚀 Memulai Batch Optimization NSGA-II ({n_runs} Runs)...")
    print(f"Populasi: 100 | Generasi: 100 | Ref Point: {ref_point}")
    
    start_total = pd.Timestamp.now()

    for i in range(1, n_runs + 1):
        run_start = pd.Timestamp.now()
        
        algorithm = NSGA2(
            pop_size=100,
            sampling=LHS(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Seed berbeda tiap run untuk validitas statistik
        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=get_termination("n_gen", 100),
            seed=i + 100,
            verbose=False
        )
        
        # Hitung Hypervolume
        if res.F is not None and len(res.F) > 0:
            current_hv = hv_calculator(res.F)
        else:
            current_hv = 0
            
        hv_results_nsga.append(current_hv)
        
        duration = (pd.Timestamp.now() - run_start).total_seconds()
        print(f"   ✅ Run {i}/{n_runs} Selesai | HV: {current_hv:.6f} | Waktu: {duration:.1f}s")

    # --- Simpan Hasil ke CSV ---
    df_hv_nsga = pd.DataFrame({
        "Run": range(1, n_runs + 1),
        "HV_NSGA": hv_results_nsga
    })
    
    path_hv_nsga = os.path.join(SCRIPT_DIR, 'hv_nsga_30_runs.csv')
    df_hv_nsga.to_csv(path_hv_nsga, index=False)
    
    total_duration = (pd.Timestamp.now() - start_total).total_seconds() / 60
    print("\n" + "="*50)
    print(f"✅ BATCH NSGA-II SELESAI!")
    print(f"📊 Mean HV: {np.mean(hv_results_nsga):.6f}")
    print(f"📁 Data HV disimpan ke: {path_hv_nsga}")
    print(f"⏱️ Total Waktu: {total_duration:.1f} Menit")