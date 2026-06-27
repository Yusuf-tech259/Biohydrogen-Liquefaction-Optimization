import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === STEP 1: Baca file hasil dengan kolom 'Generation' ===
file_path = "all_evaluated_with_generation.xlsx"
df = pd.read_excel(file_path)

# Pastikan kolom penting ada
obj_cols = ["SEC", "Efisiensi Exergy", "LCOH"]
assert all(col in df.columns for col in obj_cols), "❌ Kolom objektif tidak lengkap di file."

# Dapatkan jumlah generasi unik
generations = sorted(df["Generation"].unique())
print(f"✅ Ditemukan {len(generations)} generasi dalam data.")

# === STEP 2: Definisikan fungsi TOPSIS ===
def topsis_select(F, directions, weights=None):
    n_points, n_obj = F.shape
    if weights is None:
        weights = np.ones(n_obj)
    weights = np.array(weights, dtype=float)

    # konversi semua jadi benefit
    F2 = F.copy().astype(float)
    for j, d in enumerate(directions):
        if d.lower() == "min":
            F2[:, j] = -F2[:, j]

    # normalisasi dan pembobotan
    norm = np.linalg.norm(F2, axis=0)
    norm[norm == 0] = 1
    R = F2 / norm
    V = R * weights

    # titik ideal dan anti-ideal
    ideal = V.max(axis=0)
    neg_ideal = V.min(axis=0)

    # jarak
    d_pos = np.linalg.norm(V - ideal, axis=1)
    d_neg = np.linalg.norm(V - neg_ideal, axis=1)

    # skor TOPSIS
    score = d_neg / (d_pos + d_neg + 1e-12)
    idx = np.argmax(score)
    return idx, score

# === STEP 3: Jalankan TOPSIS untuk setiap generasi ===
directions = ["min", "max", "min"]  # urutan objektif sesuai tujuan
selected_rows = []

for g in generations:
    df_gen = df[df["Generation"] == g]
    F = df_gen[obj_cols].values
    idx, score = topsis_select(F, directions)
    best_row = df_gen.iloc[idx].copy()
    best_row["TOPSIS_Score"] = score[idx]
    selected_rows.append(best_row)

df_topsis = pd.DataFrame(selected_rows)
df_topsis.reset_index(drop=True, inplace=True)
df_topsis.to_excel("best_compromise_per_generation.xlsx", index=False)
print("✅ File 'best_compromise_per_generation.xlsx' disimpan (1 solusi terbaik per generasi).")

# === STEP 4: Visualisasi evolusi tiap objektif ===
plt.figure(figsize=(10, 8))
gens = df_topsis["Generation"]

plt.subplot(3,1,1)
plt.plot(gens, df_topsis["SEC"], '-o', color='blue')
plt.ylabel("SEC (kWh/kg)")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(gens, df_topsis["Efisiensi Exergy"], '-o', color='green')
plt.ylabel("Efisiensi Exergy (%)")
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(gens, df_topsis["LCOH"], '-o', color='red')
plt.xlabel("Generasi")
plt.ylabel("LCOH ($/kg)")
plt.grid(True)

plt.suptitle("Evolusi Solusi Kompromi (TOPSIS) per Generasi – MOGA NSGA-II", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("evolusi_topsis_per_generasi.png", dpi=300)
plt.show()

print("📊 Plot evolusi disimpan sebagai 'evolusi_topsis_per_generasi.png'")
