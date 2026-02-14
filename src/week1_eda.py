import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# ========= CONFIG CONNEXION (MySQL + SQLAlchemy) =========
USER = "root"
PASSWORD = "root"
HOST = "localhost"
PORT = 3306
DB = "erp_sales"

# mysql+mysqlconnector car on utilise mysql-connector-python
engine = create_engine(f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}")

print(" WEEK 1 - EDA sur histovente (SQLAlchemy)")
print("=" * 70)

# =========================================================
# 1) INFOS GÉNÉRALES
# =========================================================
query_info = """
SELECT 
    COUNT(*)                    AS total_lignes,
    COUNT(DISTINCT CodeArticle) AS nb_articles,
    COUNT(DISTINCT CodeMag)     AS nb_magasins,
    MIN(Date)                   AS premiere_vente,
    MAX(Date)                   AS derniere_vente
FROM histovente;
"""
info = pd.read_sql(query_info, engine)
print("\n[1] Informations générales :")
print(info.to_string(index=False))

# =========================================================
# 2) QUALITÉ DES DONNÉES
# =========================================================
query_quality = """
SELECT 
    SUM(CASE WHEN Date IS NULL        THEN 1 ELSE 0 END) AS date_null,
    SUM(CASE WHEN CodeArticle IS NULL THEN 1 ELSE 0 END) AS article_null,
    SUM(CASE WHEN Quantite IS NULL    THEN 1 ELSE 0 END) AS qte_null,
    SUM(CASE WHEN Prix IS NULL        THEN 1 ELSE 0 END) AS prix_null,
    SUM(CASE WHEN Quantite <= 0       THEN 1 ELSE 0 END) AS qte_neg_ou_zero,
    SUM(CASE WHEN Prix <= 0           THEN 1 ELSE 0 END) AS prix_neg_ou_zero,
    COUNT(*) AS total
FROM histovente;
"""
quality = pd.read_sql(query_quality, engine)
print("\n[2] Qualité des données :")
print(quality.to_string(index=False))

# =========================================================
# 3) CRÉATION D’UN ÉCHANTILLON NETTOYÉ (~10 %)
# =========================================================
print("\n[3] Création d’un échantillon nettoyé (10 %) ...")

query_sample = """
SELECT 
    Date,
    CodeMag,
    CodeArticle,
    Famille,
    Saison,
    Prix,
    Quantite,
    Total
FROM histovente
WHERE Date IS NOT NULL
  AND Quantite > 0
  AND Prix > 0
  AND RAND() < 0.10
ORDER BY Date;
"""

df = pd.read_sql(query_sample, engine)

print(f"    → lignes dans l’échantillon : {len(df):,}")
print(f"    → période : {df['Date'].min()} → {df['Date'].max()}")

# Créer une colonne revenu (au cas où Total ne serait pas toujours cohérent)
df["Revenu"] = df["Prix"] * df["Quantite"]

df.to_csv("../data/processed/sales_sample_histovente.csv", index=False)
print("    → fichier : sales_sample_histovente.csv sauvegardé.")

# =========================================================
# 4) STATISTIQUES DE BASE
# =========================================================
print("\n[4] Statistiques descriptives :")
print(df[["Quantite", "Prix", "Revenu"]].describe().to_string())

print("\nTop 10 articles par quantité vendue :")
top_articles = (
    df.groupby("CodeArticle")[["Quantite", "Revenu"]]
      .sum()
      .sort_values("Quantite", ascending=False)
      .head(10)
)
print(top_articles.to_string())

print("\nTop 10 familles par revenu :")
top_familles = (
    df.groupby("Famille")[["Quantite", "Revenu"]]
      .sum()
      .sort_values("Revenu", ascending=False)
      .head(10)
)
print(top_familles.to_string())

# =========================================================
# 5) VISUALISATIONS
# =========================================================
print("\n[5] Création des graphiques ...")
sns.set(style="whitegrid")

df["Date"] = pd.to_datetime(df["Date"])
df["Mois"] = df["Date"].dt.month
df["JourSemaine"] = df["Date"].dt.dayofweek  # 0=lundi

# 5.1 Ventes quotidiennes
daily = df.groupby("Date")["Quantite"].sum()
plt.figure(figsize=(14, 5))
plt.plot(daily.index, daily.values)
plt.title("Ventes quotidiennes (échantillon histovente)")
plt.xlabel("Date")
plt.ylabel("Quantité vendue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../reports/figures/1_ventes_quotidiennes.png")
plt.close()

# 5.2 Ventes par mois
monthly = df.groupby("Mois")["Quantite"].sum()
plt.figure(figsize=(8, 4))
monthly.plot(kind="bar")
plt.title("Ventes par mois")
plt.xlabel("Mois")
plt.ylabel("Quantité")
plt.tight_layout()
plt.savefig("../reports/figures/2_ventes_par_mois.png")
plt.close()

# 5.3 Ventes par jour de semaine
weekly = df.groupby("JourSemaine")["Quantite"].sum()
plt.figure(figsize=(8, 4))
weekly.plot(kind="bar")
plt.title("Ventes par jour de semaine (0 = lundi)")
plt.xlabel("Jour de semaine")
plt.ylabel("Quantité")
plt.tight_layout()
plt.savefig("../reports/figures/3_ventes_par_joursemaine.png")
plt.close()

# 5.4 Ventes par famille
plt.figure(figsize=(10, 5))
top_familles["Revenu"].sort_values(ascending=True).plot(kind="barh")
plt.title("Top 10 familles par revenu (échantillon)")
plt.xlabel("Revenu")
plt.tight_layout()
plt.savefig("../reports/figures/4_top_familles_revenu.png")
plt.close()

print("    → Graphiques créés :")
print("       - 1_ventes_quotidiennes.png")
print("       - 2_ventes_par_mois.png")
print("       - 3_ventes_par_joursemaine.png")
print("       - 4_top_familles_revenu.png")

print("\n WEEK1 EDA sur histovente terminé.")
