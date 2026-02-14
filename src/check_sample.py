import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/V1/sales_sample_histovente.csv", parse_dates=["Date"])
df = df.sort_values("Date")

daily = df.groupby("Date")["Quantite"].sum()

plt.figure(figsize=(14, 5))
plt.plot(daily.index, daily.values)
plt.title("Ventes quotidiennes (échantillon sauvegardé)")
plt.xlabel("Date")
plt.ylabel("Quantité")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
