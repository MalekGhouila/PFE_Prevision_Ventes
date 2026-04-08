import pandas as pd
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
    f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
    pool_timeout=300,
    pool_recycle=3600
)

print("=" * 70)
print("     FEATURE ENGINEERING - hv_daily_aggregated")
print("=" * 70)

# ========= ÉTAPE 1 : CHARGER hv_daily_aggregated =========
print("\n[ÉTAPE 1] Chargement de hv_daily_aggregated...")

df = pd.read_sql("""
    SELECT Date, CodeMag, Famille, Pays, sum_Quantite, sum_Total, nb_lignes
    FROM hv_daily_aggregated
    ORDER BY CodeMag, Famille, Date
""", engine)

df['Date'] = pd.to_datetime(df['Date'])
print(f"    Lignes chargées : {len(df):,}")
print(f"    Période         : {df['Date'].min().date()} → {df['Date'].max().date()}")

# ========= ÉTAPE 2 : FEATURES TEMPORELLES =========
print("\n[ÉTAPE 2] Ajout des features temporelles...")

df['year']           = df['Date'].dt.year
df['month']          = df['Date'].dt.month
df['day']            = df['Date'].dt.day
df['day_of_week']    = df['Date'].dt.dayofweek      # 0=Lundi, 6=Dimanche
df['week_of_year']   = df['Date'].dt.isocalendar().week.astype(int)
df['quarter']        = df['Date'].dt.quarter
df['is_weekend']     = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
df['is_month_end']   = df['Date'].dt.is_month_end.astype(int)

print(f"    ✅ Features temporelles ajoutées : year, month, day, day_of_week,")
print(f"       week_of_year, quarter, is_weekend, is_month_start, is_month_end")

# ========= ÉTAPE 3 : ENCODING PAYS + FAMILLE =========
print("\n[ÉTAPE 3] Encoding Pays et Famille...")

df['Pays_encoded']    = df['Pays'].astype('category').cat.codes
df['Famille_encoded'] = df['Famille'].astype('category').cat.codes
df['CodeMag_encoded'] = df['CodeMag'].astype('category').cat.codes

print(f"    ✅ Pays uniques     : {df['Pays'].nunique()}")
print(f"    ✅ Familles uniques : {df['Famille'].nunique()}")
print(f"    ✅ Magasins uniques : {df['CodeMag'].nunique()}")

# ========= ÉTAPE 4 : LAG FEATURES =========
print("\n[ÉTAPE 4] Calcul des lag features (J-7, J-14, J-28)...")

df = df.sort_values(['CodeMag', 'Famille', 'Date'])
group = df.groupby(['CodeMag', 'Famille'])['sum_Quantite']

df['lag_7']  = group.shift(7)
df['lag_14'] = group.shift(14)
df['lag_28'] = group.shift(28)

print(f"    ✅ lag_7, lag_14, lag_28 ajoutés")

# ========= ÉTAPE 5 : ROLLING MEAN FEATURES =========
print("\n[ÉTAPE 5] Calcul des rolling means (7j, 30j)...")

df['rolling_mean_7']  = group.transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
)
df['rolling_mean_30'] = group.transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
)

print(f"    ✅ rolling_mean_7, rolling_mean_30 ajoutés")

# ========= ÉTAPE 6 : DOMINANT ArCouleur + ArFamille =========
print("\n[ÉTAPE 6] Récupération dominant ArCouleur + ArFamille...")

dominant = pd.read_sql("""
    SELECT
        Date,
        CodeMag,
        Famille,
        ArCouleur AS dominant_couleur,
        ArFamille AS dominant_sfamille
    FROM (
        SELECT
            Date,
            CodeMag,
            Famille,
            ArCouleur,
            ArFamille,
            COUNT(*) AS nb,
            ROW_NUMBER() OVER (
                PARTITION BY Date, CodeMag, Famille
                ORDER BY COUNT(*) DESC
            ) AS rn
        FROM histovente_clean_v1
        WHERE ArCouleur IS NOT NULL AND ArCouleur != ''
        GROUP BY Date, CodeMag, Famille, ArCouleur, ArFamille
    ) ranked
    WHERE rn = 1
""", engine)

dominant['Date'] = pd.to_datetime(dominant['Date'])
df = df.merge(dominant, on=['Date', 'CodeMag', 'Famille'], how='left')

print(f"    ✅ dominant_couleur, dominant_sfamille ajoutés")

# ========= ÉTAPE 7 : NETTOYAGE FINAL =========
print("\n[ÉTAPE 7] Nettoyage final...")

rows_before = len(df)
df = df.dropna(subset=['lag_7', 'lag_14', 'lag_28'])
rows_after = len(df)

print(f"    Lignes avant dropna : {rows_before:,}")
print(f"    Lignes après dropna : {rows_after:,}")
print(f"    Lignes supprimées   : {rows_before - rows_after:,} (premières semaines sans lag)")

# ========= ÉTAPE 8 : EXPORT CSV =========
print("\n[ÉTAPE 8] Export CSV...")

output_path = "hv_features.csv"
df.to_csv(output_path, index=False)

print(f"    ✅ Fichier exporté : {output_path}")
print(f"    Lignes            : {len(df):,}")
print(f"    Colonnes          : {len(df.columns)}")
print(f"\n    Colonnes disponibles :")
for col in df.columns:
    print(f"    - {col}")

print("\n" + "=" * 70)
print("     FEATURE ENGINEERING TERMINÉ ✅")
print("=" * 70)
print(f"\n    Prochaine étape : Entraînement du modèle XGBoost")
print("=" * 70)