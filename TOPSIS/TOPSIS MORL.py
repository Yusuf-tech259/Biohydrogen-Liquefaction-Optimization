# ============================================================
# 🧠 Stability TOPSIS Analysis – MORL PPO (100 episode aggregated)
# Input : morl_agg_100points_mean.xlsx
# Output: Figure21a (evolusi objektif), Figure21b (stability ratio),
#         statistik stabilitas dalam terminal.
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === 1️⃣ Baca data hasil ringkasan ===
df = pd.read_excel("morl_agg_100points_mean.xlsx")
print(f"✅ Data MORL (aggregated) dimuat: {df.shape[0]} baris")

# === 2️⃣ Deteksi kolom numerik ===
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "Episode" in numeric_cols:
    numeric_cols.remove("Episode")

# Ambil kolom objektif dan stability ratio
obj_cols = [c for c in numeric_cols if "SEC" in c or "Exergy" in c or "LCOH" in c]
stab_col = "Stability_Ratio" if "Stability_Ratio" in df.columns else numeric_cols[-1]

print(f"Kolom objektif yang digunakan: {obj_cols}")
print(f"Kolom stabilitas yang digunakan: {stab_col}")

# === 3️⃣ Plot Figure 21a – Evolusi Solusi Kompromi (3 objektif) ===
plt.figure(figsize=(10,8))
eps = df["Episode"]

plt.subplot(3,1,1)
plt.plot(eps, df[obj_cols[0]], '-o', color='blue')
plt.ylabel(obj_cols[0]); plt.grid(True)

plt.subplot(3,1,2)
plt.plot(eps, df[obj_cols[1]], '-o', color='green')
plt.ylabel(obj_cols[1]); plt.grid(True)

plt.subplot(3,1,3)
plt.plot(eps, df[obj_cols[2]], '-o', color='red')
plt.xlabel("Aggregated Episode")
plt.ylabel(obj_cols[2]); plt.grid(True)

plt.suptitle("Figure 21a – Evolusi Solusi Kompromi (TOPSIS) – MORL PPO (Aggregated)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Fig21a_Evolusi_TOPSIS_MORL_aggregated.png", dpi=300)
plt.show()

# === 4️⃣ Plot Figure 21b – Stability Ratio ===
plt.figure(figsize=(8,4))
plt.plot(df["Episode"], df[stab_col], '-o', color='purple')
plt.xlabel("Aggregated Episode")
plt.ylabel("Stability Ratio")
plt.title("Figure 21b – Stability of TOPSIS-Selected Compromise Solution – MORL PPO (Aggregated)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Fig21b_Stability_TOPSIS_MORL_aggregated.png", dpi=300)
plt.show()

# === 5️⃣ Hitung statistik stabilitas (rata-rata, std, CV) ===
mean_stab = df[stab_col].mean()
std_stab = df[stab_col].std()
cv_stab = std_stab / mean_stab

print("\n📊 Statistik Stability TOPSIS – MORL PPO (Aggregated)")
print("---------------------------------------------------------")
print(f"Rata-rata Stability Ratio : {mean_stab:.4f}")
print(f"Standar Deviasi           : {std_stab:.4f}")
print(f"Koefisien Variasi (CV)    : {cv_stab:.4f}")
print("---------------------------------------------------------")

# === 6️⃣ Tambahan: nilai rata-rata tiap objektif ===
print("\n📈 Rata-rata nilai objektif:")
for c in obj_cols:
    print(f"- {c:<25}: {df[c].mean():.4f}")

print("\n✅ Analisis selesai!")
print("Gambar tersimpan sebagai:")
print(f"- {os.path.abspath('Fig21a_Evolusi_TOPSIS_MORL_aggregated.png')}")
print(f"- {os.path.abspath('Fig21b_Stability_TOPSIS_MORL_aggregated.png')}")
