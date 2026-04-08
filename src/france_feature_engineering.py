import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ========= CONFIG =========
USER     = "root"
PASSWORD = "root"
HOST     = "localhost"
PORT     = 3306
DB       = "erp_sales"

engine = create_engine(
    f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
    pool_timeout=300,
    pool_recycle=3600
)

print("=" * 70)
print("     FEATURE ENGINEERING - hv_france_daily")
print("=" * 70)

# ========= ÉTAPE 1 : CHARGER hv_france_daily =========
print("\n[ÉTAPE 1] Chargement de hv_france_daily...")

df = pd.read_sql("""
    SELECT Date, Famille, sum_Quantite, sum_Total, nb_magasins
    FROM hv_france_daily
    ORDER BY Famille, Date
""", engine)

df['Date'] = pd.to_datetime(df['Date'])
print(f"    Lignes chargées : {len(df):,}")

# ========= ÉTAPE 2 : NETTOYAGE FAMILLE =========
print("\n[ÉTAPE 2] Nettoyage des familles...")

before = df['Famille'].nunique()
df = df[df['Famille'].notna()]
df = df[~df['Famille'].str.startswith('~')]
df = df[df['Famille'].str.match(r'^[A-Za-zÀ-ÿ/\s]+$')]

print(f"    Familles avant nettoyage : {before}")
print(f"    Familles après nettoyage : {df['Famille'].nunique()}")
print(f"    Familles gardées : {sorted(df['Famille'].unique())}")

# ========= ÉTAPE 3 : REINDEX + FILL MISSING DAYS =========
print("\n[ÉTAPE 3] Reindex pour combler les jours manquants...")

full_date_range = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
families = df['Famille'].dropna().unique()

dfs_filled = []
for famille in families:
    df_f = df[df['Famille'] == famille].set_index('Date')
    df_f = df_f.reindex(full_date_range)
    df_f['Date']         = df_f.index
    df_f['Famille']      = famille
    df_f['sum_Quantite'] = df_f['sum_Quantite'].fillna(0)
    df_f['sum_Total']    = df_f['sum_Total'].fillna(0)
    df_f['nb_magasins']  = df_f['nb_magasins'].fillna(0)
    dfs_filled.append(df_f.reset_index(drop=True))

df = pd.concat(dfs_filled, ignore_index=True)
print(f"    Lignes après reindex : {len(df):,}")

# ========= ÉTAPE 4 : FEATURES TEMPORELLES =========
print("\n[ÉTAPE 4] Ajout des features temporelles...")

df['year']           = df['Date'].dt.year
df['month']          = df['Date'].dt.month
df['day']            = df['Date'].dt.day
df['day_of_week']    = df['Date'].dt.dayofweek
df['week_of_year']   = df['Date'].dt.isocalendar().week.astype(int)
df['quarter']        = df['Date'].dt.quarter
df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df['is_month_end']   = df['Date'].dt.is_month_end.astype(int)

print(f"    ✅ Features temporelles ajoutées")

# ========= ÉTAPE 5 : ENCODING FAMILLE =========
print("\n[ÉTAPE 5] Encoding Famille...")

df['Famille_encoded'] = df['Famille'].astype('category').cat.codes
print(f"    ✅ {df['Famille'].nunique()} familles encodées")

# ========= ÉTAPE 6 : LAG + ROLLING FEATURES =========
print("\n[ÉTAPE 6] Calcul lag + rolling features par Famille...")

df = df.sort_values(['Famille', 'Date']).reset_index(drop=True)

for famille in df['Famille'].unique():
    mask = df['Famille'] == famille
    s = df.loc[mask, 'sum_Quantite']
    df.loc[mask, 'lag_7']           = s.shift(7).values
    df.loc[mask, 'lag_14']          = s.shift(14).values
    df.loc[mask, 'lag_28']          = s.shift(28).values
    df.loc[mask, 'rolling_mean_7']  = s.shift(1).rolling(7).mean().values
    df.loc[mask, 'rolling_mean_30'] = s.shift(1).rolling(30).mean().values

print(f"    ✅ lag_7, lag_14, lag_28, rolling_mean_7, rolling_mean_30 ajoutés")

# ========= ÉTAPE 7 : NETTOYAGE FINAL =========
print("\n[ÉTAPE 7] Suppression des NaN (warmup lags)...")

before = len(df)
df = df.dropna(subset=['lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_30'])
print(f"    Supprimées : {before - len(df):,} lignes")
print(f"    Lignes finales : {len(df):,}")

# ========= ÉTAPE 8 : EXPORT =========
print("\n[ÉTAPE 8] Export CSV...")

df.to_csv("hv_france_features.csv", index=False)

print(f"    ✅ Exporté : hv_france_features.csv")
print(f"    Lignes    : {len(df):,}")
print(f"    Familles  : {df['Famille'].nunique()}")
print(f"    Période   : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"    Colonnes  : {list(df.columns)}")

print("\n" + "=" * 70)
print("     FEATURE ENGINEERING TERMINÉ ✅")
print("=" * 70)