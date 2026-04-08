import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

USER, PASSWORD, HOST, PORT, DB = "root", "root", "localhost", 3306, "erp_sales"
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed", "V2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  FEATURE ENGINEERING V2 — hv_france_weekly")
print("=" * 60)

# ===== LOAD =====
print("\n[1] Loading hv_france_weekly...")
df = pd.read_sql("SELECT * FROM hv_france_weekly ORDER BY Famille, year, week_of_year", engine)
df['week_start'] = pd.to_datetime(df['week_start'])
print(f"    Rows: {len(df):,} | Families: {df['Famille'].nunique()}")

# ===== CALENDAR FEATURES =====
print("\n[2] Calendar features...")
df['month']     = df['week_start'].dt.month
df['quarter']   = df['week_start'].dt.quarter
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
df['is_soldes'] = df['month'].isin([1, 7]).astype(int)

# ===== HIGH SEASON INTERACTION =====
print("\n[3] Famille × Season interaction feature...")
df['is_high_season'] = 0

summer_fam = ['BERMUDA', 'BERMUDA/SHORT', 'SHORT', 'T-SHIRT',
              'TEE-SHIRT', 'T SHIRT SANS MANCHES', 'T SHIRT MANCHES COUR']
winter_fam = ['MANTEAU', 'PULL', 'BONNETERIE', 'BONNETERIE/COIFFANT',
              'GILET', 'SWEAT', 'SWEATSHIRT']

for fam in summer_fam:
    mask = (df['Famille'] == fam) & (df['is_summer'] == 1)
    df.loc[mask, 'is_high_season'] = 1

for fam in winter_fam:
    mask = (df['Famille'] == fam) & (df['is_winter'] == 1)
    df.loc[mask, 'is_high_season'] = 1

print(f"    ✅ is_high_season added")

# ===== ENCODE FAMILLE =====
print("\n[4] Encoding Famille...")
df['Famille_encoded'] = df['Famille'].astype('category').cat.codes
print(f"    ✅ {df['Famille'].nunique()} familles encodées")

# ===== LAG + ROLLING =====
print("\n[5] Lag & rolling features...")
df = df.sort_values(['Famille', 'year', 'week_of_year']).reset_index(drop=True)

for famille in df['Famille'].unique():
    mask = df['Famille'] == famille
    s = df.loc[mask, 'sum_Quantite']
    df.loc[mask, 'lag_1']           = s.shift(1).values
    df.loc[mask, 'lag_2']           = s.shift(2).values
    df.loc[mask, 'lag_4']           = s.shift(4).values
    df.loc[mask, 'lag_52']          = s.shift(52).values
    df.loc[mask, 'rolling_mean_4']  = s.shift(1).rolling(4).mean().values
    df.loc[mask, 'rolling_mean_12'] = s.shift(1).rolling(12).mean().values

print(f"    ✅ lag_1, lag_2, lag_4, lag_52, rolling_mean_4, rolling_mean_12")

# ===== DROPNA =====
print("\n[6] Dropping NaN warmup rows...")
before = len(df)

# Drop only rows where core features are missing (first ~12 weeks per family)
core_cols = ['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_mean_12']
df = df.dropna(subset=core_cols)

# lag_52 NaN = first year of data → fill with rolling_mean_12 as approximation
lag52_missing = df['lag_52'].isna().sum()
df['lag_52'] = df['lag_52'].fillna(df['rolling_mean_12'])

print(f"    Dropped          : {before - len(df):,}")
print(f"    lag_52 filled    : {lag52_missing:,} rows (used rolling_mean_12)")
print(f"    Remaining        : {len(df):,}")
print(f"    Date range       : {df['week_start'].min().date()} → {df['week_start'].max().date()}")

# ===== EXPORT =====
print("\n[7] Exporting...")
output_path = os.path.join(OUTPUT_DIR, "hv_france_weekly_features_v2.csv")
df.to_csv(output_path, index=False)
print(f"    ✅ Exported to: {output_path}")
print(f"    Rows     : {len(df):,}")
print(f"    Families : {df['Famille'].nunique()}")
print(f"    Columns  : {list(df.columns)}")
