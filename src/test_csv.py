import pandas as pd
df = pd.read_csv("hv_features.csv")
print(f"Total lignes : {len(df)}")
print(f"Colonnes : {df.columns.tolist()}")
print(df['Famille'].value_counts().head(10))
print(df['Pays'].value_counts())
print(df['Date'].min(), "→", df['Date'].max())