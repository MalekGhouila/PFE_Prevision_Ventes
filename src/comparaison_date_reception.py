import pandas as pd
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

# ========= CONFIG CONNEXION =========
USER = "root"
PASSWORD = "root"
HOST = "localhost"
PORT = 3306
DB = "erp_sales"

engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

print("=" * 70)
print("     COMPARAISON DATE vs RECEPTION - HISTOVENTE")
print("=" * 70)

resultats = {}

# ================================================
# 1. Vue d'ensemble : NULL dans chaque colonne
# ================================================
print("\n[1/6] Analyse des valeurs NULL...")

q1 = """
SELECT 
    COUNT(*) as total_lignes,
    SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) as date_null,
    ROUND(SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_date_null,
    SUM(CASE WHEN Reception IS NULL THEN 1 ELSE 0 END) as reception_null,
    ROUND(SUM(CASE WHEN Reception IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_reception_null
FROM histovente;
"""
df1 = pd.read_sql(q1, engine)
resultats['1_NULL_Overview'] = df1
print(df1.to_string(index=False))

# ================================================
# 2. Combien de lignes où Date ≠ Reception ?
# ================================================
print("\n[2/6] Lignes où Date ≠ Reception...")

q2 = """
SELECT 
    COUNT(*) as lignes_differentes,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM histovente), 2) as pct_differentes
FROM histovente
WHERE Date IS NOT NULL 
AND Reception IS NOT NULL
AND Date != Reception;
"""
df2 = pd.read_sql(q2, engine)
resultats['2_Lignes_Differentes'] = df2
print(df2.to_string(index=False))

# ================================================
# 3. Écart moyen en jours entre Date et Reception
# ================================================
print("\n[3/6] Écart en jours entre Date et Reception...")

q3 = """
SELECT 
    AVG(ABS(DATEDIFF(Date, Reception))) as ecart_moyen_jours,
    MIN(DATEDIFF(Date, Reception)) as ecart_min_jours,
    MAX(DATEDIFF(Date, Reception)) as ecart_max_jours
FROM histovente
WHERE Date IS NOT NULL 
AND Reception IS NOT NULL;
"""
df3 = pd.read_sql(q3, engine)
resultats['3_Ecart_Jours'] = df3
print(df3.to_string(index=False))

# ================================================
# 4. Distribution des écarts
# ================================================
print("\n[4/6] Distribution des écarts (Top 20)...")

q4 = """
SELECT 
    DATEDIFF(Date, Reception) as ecart_jours,
    COUNT(*) as nb_lignes,
    ROUND(COUNT(*) * 100.0 / (
        SELECT COUNT(*) FROM histovente 
        WHERE Date IS NOT NULL AND Reception IS NOT NULL
    ), 2) as pourcentage
FROM histovente
WHERE Date IS NOT NULL 
AND Reception IS NOT NULL
GROUP BY DATEDIFF(Date, Reception)
ORDER BY nb_lignes DESC
LIMIT 20;
"""
df4 = pd.read_sql(q4, engine)
resultats['4_Distribution_Ecarts'] = df4
print(df4.to_string(index=False))

# ================================================
# 5. Exemples de lignes où Date ≠ Reception
# ================================================
print("\n[5/6] Exemples de lignes où Date ≠ Reception...")

q5 = """
SELECT 
    IDHistoVente,
    Date,
    Reception,
    DATEDIFF(Date, Reception) as ecart_jours,
    TypeVente,
    Famille,
    Prix
FROM histovente
WHERE Date IS NOT NULL 
AND Reception IS NOT NULL
AND Date != Reception
LIMIT 20;
"""
df5 = pd.read_sql(q5, engine)
resultats['5_Exemples_Differences'] = df5
print(df5.to_string(index=False))

# ================================================
# 6. Qualité de Reception par année
# ================================================
print("\n[6/6] Qualité par année...")

q6 = """
SELECT 
    YEAR(Reception) as annee,
    COUNT(*) as total_lignes,
    SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) as date_null,
    ROUND(SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_date_null
FROM histovente
WHERE Reception >= '2021-01-01'
AND Reception <= '2025-12-31'
GROUP BY YEAR(Reception)
ORDER BY annee;
"""
df6 = pd.read_sql(q6, engine)
resultats['6_Qualite_Par_Annee'] = df6
print(df6.to_string(index=False))

# ================================================
# SAUVEGARDE EXCEL
# ================================================
print("\n" + "=" * 70)
print("     SAUVEGARDE DES RÉSULTATS")
print("=" * 70)

output_file = "comparaison_date_reception.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name, df in resultats.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\n✅ Fichier créé : {output_file}")
print("\n📊 Onglets générés :")
for sheet in resultats.keys():
    print(f"   - {sheet}")

print("\n" + "=" * 70)
print("     ANALYSE TERMINÉE")
print("=" * 70)
