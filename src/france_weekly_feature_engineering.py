import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

USER, PASSWORD, HOST, PORT, DB = "root", "root", "localhost", 3306, "erp_sales"
engine = create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

print("=" * 60)
print("  FEATURE ENGINEERING — hv_france_weekly")
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

# ===== ENCODE FAMILLE =====
print("\n[3] Encoding Famille...")
df['Famille_encoded'] = df['Famille'].astype('category').cat.codes
print(f"    ✅ {df['Famille'].nunique()} familles encodées")

# ===== LAG + ROLLING (per Famille) =====
print("\n[4] Lag & rolling features...")
df = df.sort_values(['Famille', 'year', 'week_of_year']).reset_index(drop=True)

for famille in df['Famille'].unique():
    mask = df['Famille'] == famille
    s = df.loc[mask, 'sum_Quantite']
    df.loc[mask, 'lag_1']           = s.shift(1).values   # last week
    df.loc[mask, 'lag_2']           = s.shift(2).values   # 2 weeks ago
    df.loc[mask, 'lag_4']           = s.shift(4).values   # 1 month ago
    # lag_52 REMOVED — was consuming all 2022 data as warmup
    df.loc[mask, 'rolling_mean_4']  = s.shift(1).rolling(4).mean().values
    df.loc[mask, 'rolling_mean_12'] = s.shift(1).rolling(12).mean().values

print(f"    ✅ lag_1, lag_2, lag_4, rolling_mean_4, rolling_mean_12 added")
print(f"    ℹ️  lag_52 removed (was consuming full year as warmup)")

# ===== DROPNA =====
print("\n[5] Dropping NaN warmup rows...")
before = len(df)
df = df.dropna()
print(f"    Dropped : {before - len(df):,}")
print(f"    Remaining: {len(df):,}")
print(f"    Date range: {df['week_start'].min().date()} → {df['week_start'].max().date()}")

# ===== EXPORT =====
print("\n[6] Exporting...")
df.to_csv("hv_france_weekly_features.csv", index=False)
print(f"    ✅ hv_france_weekly_features.csv")
print(f"    Rows     : {len(df):,}")
print(f"    Families : {df['Famille'].nunique()}")
print(f"    Columns  : {list(df.columns)}")