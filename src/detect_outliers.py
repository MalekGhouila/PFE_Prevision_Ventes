import pandas as pd

df = pd.read_csv("../data/processed/V1/sales_sample_histovente.csv", parse_dates=["Date"])
daily = df.groupby("Date")["Quantite"].sum().reset_index()

# Trier par Quantite décroissante
daily_sorted = daily.sort_values("Quantite", ascending=False)
print(daily_sorted.head(10))
