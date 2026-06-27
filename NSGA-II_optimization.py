import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.decomposition.asf import ASF

# =====================
# === KONFIGURASI ===
# =====================

# Mendapatkan lokasi folder script ini berada agar pembacaan file aman
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Nama File Model & Scaler
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_ann_surrogate_model_LH2_FINAL.keras')
SCALER_X_PATH = os.path.join(SCRIPT_DIR, 'x_scaler.gz')
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, 'y_scaler.gz')

# --- Bagian 1: Memuat Surrogate Model dan Scaler ---
print("="*50)
print(f"Memuat file pendukung dari: {SCRIPT_DIR}")

try:
    SURROGATE_MODEL = tf.keras.models.load_model(MODEL_PATH)
    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y = joblib.load(SCALER_Y_PATH)
    print("✅ Model dan scaler berhasil dimuat.")
except Exception as e:
    raise IOError(f"GAGAL MEMUAT FILE. Pastikan file .keras dan .gz ada di folder yang sama.\nError: {e}")

# --- Bagian 2: Mendefinisikan Masalah Optimasi untuk MOGA ---

class SurrogateProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=3, n_obj=3, n_constr=0,
            # Batas Bawah [Pout_K21, Pout_VLV101, Pout_X121]
            xl=np.array([300, 110, 120]), 
            
            # Batas Atas [Pout_K21, Pout_VLV101, Pout_X121]
            xu=np.array([700, 150, 150])   
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Gunakan model surrogate untuk prediksi
        input_scaled = SCALER_X.transform(x)
        pred_scaled = SURROGATE_MODEL.predict(input_scaled, verbose=0)
        responses = SCALER_Y.inverse_transform(pred_scaled)
        
        # Urutan Output (dari y_scaler.gz): 
        # 0: SEC (kWh/kg)
        # 1: Efisiensi Exergy (%)
        # 2: LCOH (USD/kg)
        
        # Tujuan Optimasi PyMoo (Selalu Minimasi):
        # 1. SEC -> Minimalkan (Positif)
        # 2. Efisiensi Exergy -> Maksimalkan (Jadi Negatifkan)
        # 3. LCOH -> Minimalkan (Positif)
        
        out["F"] = np.column_stack([responses[:, 0], -responses[:, 1], responses[:, 2]])
        
# --- Bagian 3: Konfigurasi dan Menjalankan Algoritma Genetika ---

if __name__ == '__main__':
    problem = SurrogateProblem()
    algorithm = NSGA2(
        pop_size=100,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 100) 
    
    print("\n" + "="*50)
    print("Memulai optimisasi dengan NSGA-II (100 populasi, 100 generasi)...")
    
    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=1,
        save_history=True,
        verbose=True
    )
    print("✅ Optimisasi MOGA selesai.")

    # --- Bagian 4: Analisis dan Visualisasi Hasil ---
    print("\n" + "="*50)
    print("--- Analisis Hasil Pareto Front ---")
    
    output_column_names = ['SEC (kWh/kg)', 'Efisiensi Exergy (%)', 'LCOH (USD/kg)'] 
    input_column_names = ['Pout_K21 (kPa)', 'Pout_VLV101 (kPa)', 'Pout_X121 (kPa)']
    
    # --- Ekstraksi Data Pareto Front ---
    pareto_F_raw = res.F
    pareto_X = res.X
    pareto_F = np.copy(pareto_F_raw)
    
    # Kembalikan nilai Efisiensi Exergy (objektif ke-2) ke positif agar mudah dibaca manusia
    pareto_F[:, 1] *= -1 
    
    df_pareto_solutions = pd.concat([
        pd.DataFrame(pareto_X, columns=input_column_names),
        pd.DataFrame(pareto_F, columns=output_column_names)
    ], axis=1)
    
    print(f"\n✅ Ditemukan {len(df_pareto_solutions)} solusi Pareto Optimal.")
    
    # Simpan hasil pareto
    file_pareto = os.path.join(SCRIPT_DIR, 'pareto_front_solutions_final.xlsx')
    df_pareto_solutions.to_excel(file_pareto, index=False)
    print(f"✅ Data solusi Pareto Optimal disimpan ke: {file_pareto}")

    # --- Ekstraksi SEMUA Data yang Dievaluasi ---
    all_X = np.vstack([gen.pop.get("X") for gen in res.history])
    all_F_raw = np.vstack([gen.pop.get("F") for gen in res.history])
    all_F = np.copy(all_F_raw)
    all_F[:, 1] *= -1 # Kembalikan nilai Efisiensi Exergy ke positif

    df_all_solutions = pd.concat([
        pd.DataFrame(all_X, columns=input_column_names),
        pd.DataFrame(all_F, columns=output_column_names)
    ], axis=1)
    
    # Simpan semua hasil evaluasi
    file_all = os.path.join(SCRIPT_DIR, 'all_evaluated_solutions_final.xlsx')
    df_all_solutions.to_excel(file_all, index=False)
    print(f"✅ Data semua {len(df_all_solutions)} solusi disimpan ke: {file_all}")
    
    # --- Visualisasi 1: Pair Plot Informatif ---
    print("\nMembuat Pair Plot informatif (Pareto vs Semua Solusi)...")
    try:
        g = sns.PairGrid(df_all_solutions[output_column_names], corner=True)
        g.fig.suptitle("Pareto Front vs. Ruang Pencarian (Revisi)", y=1.03, fontsize=16, fontweight='bold')
        g.map_lower(sns.scatterplot, s=15, color='grey', alpha=0.2)
        g.data = df_pareto_solutions[output_column_names]
        g.map_lower(sns.scatterplot, s=40, color='#0044BB', edgecolor='white', linewidth=0.5)
        g.map_diag(sns.kdeplot, fill=True, color='#0044BB')
        plt.savefig(os.path.join(SCRIPT_DIR, 'moga_pair_plot_final.png'), dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Gagal membuat Pair Plot: {e}")

    # --- Visualisasi 2 & 3: Plot Interaktif ---
    print("\nMembuat visualisasi interaktif...")

    # -- Visualisasi 3D dengan warna berdasarkan parameter input --
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df_pareto_solutions['SEC (kWh/kg)'], y=df_pareto_solutions['Efisiensi Exergy (%)'], z=df_pareto_solutions['LCOH (USD/kg)'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_pareto_solutions['Pout_K21 (kPa)'],
            colorscale='plasma',
            opacity=0.8,
            colorbar=dict(title='Pout_K21 (kPa)')
        )
    )])
    fig_3d.update_layout(
        title='<b>Pareto Front 3D (Warna berdasarkan Pout_K21)</b>',
        scene=dict(xaxis_title='SEC (kWh/kg) (Min)', yaxis_title='Efisiensi Exergy (%) (Max)', zaxis_title='LCOH (USD/kg) (Min)')
    )
    fig_3d.write_html(os.path.join(SCRIPT_DIR, "moga_3d_pareto_front_final.html"))

    # -- Visualisasi Parallel Coordinate dengan INPUT dan OUTPUT --
    dimensions = [
        dict(label='Pout_K21 (kPa)', values=df_pareto_solutions['Pout_K21 (kPa)']),
        dict(label='Pout_VLV101 (kPa)', values=df_pareto_solutions['Pout_VLV101 (kPa)']),
        dict(label='Pout_X121 (kPa)', values=df_pareto_solutions['Pout_X121 (kPa)']),
        dict(label='SEC (Min)', values=df_pareto_solutions['SEC (kWh/kg)']),
        dict(label='Efisiensi Exergy (Max)', values=df_pareto_solutions['Efisiensi Exergy (%)']),
        dict(label='LCOH (Min)', values=df_pareto_solutions['LCOH (USD/kg)'])
    ]
    fig_par = go.Figure(data=go.Parcoords(
        line = dict(color = df_pareto_solutions['SEC (kWh/kg)'], colorscale = 'plasma', showscale = True, colorbar = {'title': 'SEC (kWh/kg)'}),
        dimensions = dimensions
    ))
    fig_par.update_layout(title={'text': "<b>Trade-off Solusi Pareto Optimal (Input & Output)</b>", 'x':0.5})
    fig_par.write_html(os.path.join(SCRIPT_DIR, "moga_parallel_plot_final.html"))

    print("✅ Visualisasi 3D dan Parallel Coordinate informatif selesai.")

    # --- Bagian 5: Analisis Pemilihan Solusi - Menemukan "Knee Point" ---
    print("\n" + "="*50)
    print("--- Analisis Titik Kompromi (Knee Point) ---")
    
    # Normalisasi data objektif mentah dari pymoo
    F_normalized = (pareto_F_raw - pareto_F_raw.min(axis=0)) / (pareto_F_raw.max(axis=0) - pareto_F_raw.min(axis=0))
    
    # Temukan knee point
    decomp = ASF()
    try:
        knee_index = decomp.do(F_normalized, 1.0/F_normalized.shape[1]).argmin()
        knee_solution = df_pareto_solutions.iloc[knee_index]
        
        print("✅ Solusi Kompromi Terbaik (Knee Point) ditemukan:")
        print(knee_solution.to_string())
        
        # Sorot Knee Point pada plot 3D
        fig_3d.add_trace(go.Scatter3d(
            x=[knee_solution['SEC (kWh/kg)']], y=[knee_solution['Efisiensi Exergy (%)']], z=[knee_solution['LCOH (USD/kg)']],
            mode='markers',
            marker=dict(color='red', size=14, symbol='x', line=dict(width=3)),
            name='Knee Point (Kompromi)'
        ))
        fig_3d.update_layout(title='<b>Pareto Front 3D dengan Solusi Kompromi (Knee Point)</b>')
        fig_3d.write_html(os.path.join(SCRIPT_DIR, "moga_3d_pareto_with_knee_final.html"))
        print("\n✅ Visualisasi 3D dengan Knee Point disimpan.")
    except Exception as e:
        print(f"Gagal mencari Knee Point (mungkin solusi terlalu sedikit): {e}")

    print("\n" + "="*50)
    print("🎉 Semua proses selesai.")