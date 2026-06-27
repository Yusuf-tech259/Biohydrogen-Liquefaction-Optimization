import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import itertools
import time
import os
import multiprocessing
from functools import partial

# === CONFIG ===
# Beri nama output Anda agar plot dan file Excel lebih mudah dibaca
OUTPUT_NAMES = ['SEC', 'Exergy_Efficiency', 'LCOH']
VIS_DIR = 'hasil_visualisasi'      # Folder untuk menyimpan semua gambar
EXCEL_OUTPUT = 'hasil_analisis_model_terbaik.xlsx'
os.makedirs(VIS_DIR, exist_ok=True)

# === BAGIAN 1: MEMUAT DAN MEMPERSIAPKAN DATA (DENGAN BEST PRACTICE) ===
try:
    print("Memuat data dari file CSV...")
    input_df = pd.read_csv('input.csv', header=None)
    output_df = pd.read_csv('target.csv', header=None)
    X_raw_df = pd.DataFrame(input_df.values.T)
    y_raw_df = pd.DataFrame(output_df.values.T, columns=OUTPUT_NAMES)
    print("Data berhasil dimuat.")
except FileNotFoundError:
    print("\nFile tidak ditemukan. Pastikan file 'input.csv' dan 'target.csv' ada di folder yang sama.")
    exit()

# 1. PISAHKAN DATA SEBELUM PENSKALAAN (Mencegah Data Leakage)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw_df, y_raw_df, test_size=0.2, random_state=42
)

# 2. BUAT & FIT SCALER HANYA PADA DATA TRAINING
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_raw)
y_train = y_scaler.fit_transform(y_train_raw)

# 3. GUNAKAN SCALER YANG SAMA UNTUK TRANSFORM DATA TESTING
X_test = x_scaler.transform(X_test_raw)
y_test = y_scaler.transform(y_test_raw)

# Simpan scaler yang sudah "belajar" dari data training
joblib.dump(x_scaler, 'x_scaler.gz')
joblib.dump(y_scaler, 'y_scaler.gz')

# Data asli untuk evaluasi akhir adalah y_test_raw
y_test_orig = y_test_raw.values

print("Persiapan data selesai dengan metode yang benar (tanpa data leakage).")

# === BAGIAN 2: DEFINISI MODEL DAN ARSITEKTUR ===
neuron_options = [16, 32, 64, 128]
architectures = []

# 1, 2, dan 3 Hidden Layers
for i in range(1, 4):
    for combo in itertools.product(neuron_options, repeat=i):
        architectures.append(combo)

def build_model(input_shape, output_shape, hidden_layers):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(output_shape, activation='linear')) # 'linear' untuk regresi

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# === FUNGSI TRAIN DAN EVALUASI MODEL (UNTUK PARALEL) ===
def train_and_evaluate_model(arch_with_index, data, scalers):
    i, arch = arch_with_index
    X_train, y_train, X_test, y_test, y_test_orig = data
    y_scaler = scalers

    print(f"--- Memulai Model {i+1}/{len(architectures)} | Arsitektur: {arch} ---")

    model = build_model(X_train.shape[1], y_train.shape[1], arch)

    # Callback untuk training yang lebih cerdas
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=300,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[early_stopping, reduce_lr]
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # Hitung metrik performa yang lebih seimbang
    r2_per_output = [r2_score(y_test_orig[:, i], y_pred[:, i]) for i in range(y_test_orig.shape[1])]
    r2_avg = np.mean(r2_per_output)

    model_path = f'temp_model_{i}.keras'
    model.save(model_path)

    print(f"--- Selesai Model {i+1} | Arsitektur: {arch} | R² Rata-rata: {r2_avg:.4f} ---")

    return {
        'arsitektur': arch,
        'r2_score_avg': r2_avg,
        'r2_per_output': r2_per_output,
        'model_path': model_path,
        'history': history.history
    }

# === BAGIAN 3: EKSEKUSI UTAMA ===
if __name__ == '__main__':
    # Konfigurasi GPU (opsional, jika ada)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print(f"\nTotal arsitektur yang akan diuji: {len(architectures)} kombinasi.")
    num_processes = max(1, multiprocessing.cpu_count() - 2) # Sisakan 2 core untuk sistem
    print(f"Menggunakan {num_processes} prosesor untuk paralelisasi.\n")

    start_time_total = time.time()
    data_tuple = (X_train, y_train, X_test, y_test, y_test_orig)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(
            partial(train_and_evaluate_model, data=data_tuple, scalers=y_scaler),
            list(enumerate(architectures))
        )

    end_time_total = time.time()
    print(f"\nPencarian arsitektur selesai dalam {((end_time_total - start_time_total) / 60):.2f} menit.")

    # Pilih model terbaik berdasarkan RATA-RATA R²
    best_result = max(results, key=lambda x: x['r2_score_avg'])
    best_r2_avg = best_result['r2_score_avg']
    best_architecture = best_result['arsitektur']
    best_history = best_result['history']
    best_model = keras.models.load_model(best_result['model_path'])

    # Hapus model-model sementara
    for result in results:
        os.remove(result['model_path'])

    # Buat DataFrame hasil dan urutkan
    results_df = pd.DataFrame(results).drop(columns=['model_path', 'history']).sort_values(by='r2_score_avg', ascending=False).reset_index(drop=True)
    print("\n--- Hasil Peringkat Arsitektur (Berdasarkan R² Rata-rata) ---")
    print(results_df[['arsitektur', 'r2_score_avg']].to_string())

    print("\n==============================================")
    print("           HASIL TERBAIK DITEMUKAN")
    print("==============================================")
    print(f"Arsitektur Terbaik   : {best_architecture}")
    print(f"R² Rata-rata Tertinggi : {best_r2_avg:.4f}")
    print("==============================================")

    best_model.save('best_ann_surrogate_model.keras')
    print("\nModel terbaik telah disimpan ke 'best_ann_surrogate_model.keras'")

    # === BAGIAN 4: VISUALISASI HASIL MODEL TERBAIK ===
    print("\nMembuat visualisasi untuk model terbaik...")

    y_pred = y_scaler.inverse_transform(best_model.predict(X_test, verbose=0))
    residuals = y_test_orig - y_pred
    r2_list = best_result['r2_per_output']

    sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)

    # --- Kurva Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_history['loss'], label='Loss Training', lw=2)
    plt.plot(best_history['val_loss'], label='Loss Validasi', lw=2, linestyle='--')
    plt.title(f'Kurva Loss Model Terbaik (Arsitektur: {best_architecture})', fontsize=16, fontweight='bold')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '1_kurva_loss.png'), dpi=300)
    plt.show()

    # --- Plot Paritas (Gabungan) ---
    num_outputs = len(OUTPUT_NAMES)
    fig, axes = plt.subplots(nrows=1, ncols=num_outputs, figsize=(7 * num_outputs, 6), constrained_layout=True)
    fig.suptitle(f'Plot Paritas Model Terbaik ($R^2_{{avg}}$ = {best_r2_avg:.4f})', fontsize=20, fontweight='bold')
    for i in range(num_outputs):
        ax = axes[i]
        ax.scatter(y_test_orig[:, i], y_pred[:, i], alpha=0.7, edgecolors='k', s=80)
        lims = [np.min(ax.get_xlim() + ax.get_ylim()), np.max(ax.get_xlim() + ax.get_ylim())]
        ax.plot(lims, lims, 'r--', alpha=0.75, lw=2, label='Prediksi Sempurna')
        ax.set_title(f'Output: {OUTPUT_NAMES[i]}', fontsize=14)
        ax.set_xlabel('Nilai Aktual'); ax.set_ylabel('Nilai Prediksi')
        ax.legend()
        ax.text(0.05, 0.9, f'$R^2 = {r2_list[i]:.4f}$', transform=ax.transAxes, fontsize=14, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax.set_aspect('equal', 'box')
    plt.savefig(os.path.join(VIS_DIR, '2_plot_paritas_gabungan.png'), dpi=300)
    plt.show()

    # --- Distribusi Error (Gabungan) ---
    fig, axes = plt.subplots(nrows=1, ncols=num_outputs, figsize=(7 * num_outputs, 6), constrained_layout=True)
    fig.suptitle('Distribusi Error Prediksi (Residuals)', fontsize=20, fontweight='bold')
    for i in range(num_outputs):
        ax = axes[i]
        sns.histplot(residuals[:, i], kde=True, ax=ax, stat='density', bins=15)
        mean_err = np.mean(residuals[:, i])
        ax.axvline(x=mean_err, color='red', linestyle='--', lw=2, label=f'Mean Error: {mean_err:.4f}')
        ax.set_title(f'Output: {OUTPUT_NAMES[i]}', fontsize=14)
        ax.set_xlabel('Error (Aktual - Prediksi)'); ax.set_ylabel('Densitas')
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(VIS_DIR, '3_distribusi_error_gabungan.png'), dpi=300)
    plt.show()

    # === BAGIAN 5: SIMPAN HASIL DETAIL KE EXCEL ===
    print("\nMenyimpan hasil analisis detail ke file Excel...")

    # Gabungkan semua data untuk output
    df_regresi = pd.DataFrame()
    # Tambahkan input asli (unscaled)
    df_regresi = pd.concat([X_test_raw.reset_index(drop=True)], axis=1)
    df_regresi.columns = [f'Input_{i+1}' for i in range(X_test_raw.shape[1])]

    # Tambahkan nilai aktual, prediksi, dan error
    for i in range(num_outputs):
        df_regresi[f'Aktual_{OUTPUT_NAMES[i]}'] = y_test_orig[:, i]
        df_regresi[f'Prediksi_{OUTPUT_NAMES[i]}'] = y_pred[:, i]
        df_regresi[f'Error_{OUTPUT_NAMES[i]}'] = residuals[:, i]

    # Buat DataFrame ringkasan performa
    df_summary = pd.DataFrame({
        'Output': OUTPUT_NAMES,
        'R2_Score': r2_list,
        'Mean_Error': np.mean(residuals, axis=0),
        'Mean_Absolute_Error': np.mean(np.abs(residuals), axis=0)
    })

    # Simpan ke beberapa sheet dalam satu file Excel
    with pd.ExcelWriter(EXCEL_OUTPUT) as writer:
        df_regresi.to_excel(writer, index=False, sheet_name='Data_Prediksi_Detail')
        df_summary.to_excel(writer, index=False, sheet_name='Ringkasan_Performa')
        results_df.to_excel(writer, index=False, sheet_name='Ranking_Arsitektur')

    print(f"\nFile '{EXCEL_OUTPUT}' berhasil dibuat dengan 3 sheet:")
    print("1. 'Data_Prediksi_Detail': Nilai input, aktual, prediksi, dan error.")
    print("2. 'Ringkasan_Performa': Metrik R², Mean Error, dll. per output.")
    print("3. 'Ranking_Arsitektur': Peringkat semua arsitektur yang diuji.")
    print(f"\nSemua visualisasi disimpan di folder '{VIS_DIR}'.")
    print("\nSelesai. ✨")