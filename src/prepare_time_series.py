import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Charger l'échantillon sauvegardé
df = pd.read_csv("../data/processed/V1/sales_sample_histovente.csv", parse_dates=["Date"])

# Agrégation quotidienne
daily = df.groupby("Date")["Quantite"].sum().reset_index()

# Trier par date
daily = daily.sort_values("Date")

print(" Série quotidienne brute :")
print(daily.describe())

# Identifier le 99e percentile (seuil d'outlier)
seuil = daily["Quantite"].quantile(0.99)
print(f"\nSeuil 99e percentile : {seuil:.0f}")

# Créer une version capée pour visualisation et éventuellement modélisation
daily["Quantite_capped"] = np.where(daily["Quantite"] > seuil, seuil, daily["Quantite"])

# Sauvegarder les deux versions
daily.to_csv("../data/processed/daily_series_raw.csv", index=False)
daily[["Date", "Quantite_capped"]].to_csv("../data/processed/daily_series_capped.csv", index=False)

print("\n Séries sauvegardées :")
print("  - daily_series_raw.csv")
print("  - daily_series_capped.csv")

# Visualisation brute
plt.figure(figsize=(14, 4))
plt.plot(daily["Date"], daily["Quantite"], label="Brut")
plt.title("Ventes quotidiennes (brut, avec outliers)")
plt.xlabel("Date")
plt.ylabel("Quantité")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../reports/figures/ts_brut.png")
plt.close()

# Visualisation capée
plt.figure(figsize=(14, 4))
plt.plot(daily["Date"], daily["Quantite_capped"], label="Capé 99e percentile")
plt.title("Ventes quotidiennes (outliers capés au 99e percentile)")
plt.xlabel("Date")
plt.ylabel("Quantité (capée)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../reports/figures/ts_capped.png")
plt.close()

print("\n Graphiques générés : ts_brut.png, ts_capped.png")
