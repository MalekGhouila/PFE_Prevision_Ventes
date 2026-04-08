import pandas as pd
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# ========= CONFIG CONNEXION =========
USER = "root"
PASSWORD = "root"
HOST = "localhost"
PORT = 3306
DB = "erp_sales"

engine = create_engine(
    f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
    pool_timeout=300,
    pool_recycle=3600
)

print("=" * 70)
print("     DATA CLEANING - HISTOVENTE → histovente_clean_v1")
print("=" * 70)

# ========= ÉTAPE 1 : COLONNES À SUPPRIMER =========
print("\n[ÉTAPE 1] Suppression des colonnes inutiles...")

colonnes_a_supprimer = [
    # >60% NULL
    'Observation',
    'LigneTicket',
    'CodeSaison',
    'ArSaison',
    'DiscountPercentage',
    'DiscountAmount',
    'TotalHT',
    'ArCode',
    'Article',
    'VATAmount',
    'StoreCategory',
    'idArSaison',
    # Techniques / inutiles
    'isSynchronised',
    'IDFactureDiva',
    'isFacture',
    'ligne',
    'Defecttrt',
    'IDLigneTicketClient'
]

print(f"    Colonnes à supprimer : {len(colonnes_a_supprimer)}")
for col in colonnes_a_supprimer:
    print(f"    - {col}")

# ========= ÉTAPE 2 : CONSTRUIRE LA REQUÊTE =========
print("\n[ÉTAPE 2] Construction de la requête de nettoyage...")

query_cols = """
SELECT COLUMN_NAME 
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_SCHEMA = 'erp_sales' 
AND TABLE_NAME = 'histovente'
ORDER BY ORDINAL_POSITION;
"""
df_cols = pd.read_sql(query_cols, engine)
toutes_colonnes = df_cols['COLUMN_NAME'].tolist()

colonnes_a_garder = [col for col in toutes_colonnes if col not in colonnes_a_supprimer]

print(f"    Total colonnes histovente : {len(toutes_colonnes)}")
print(f"    Colonnes supprimées       : {len(colonnes_a_supprimer)}")
print(f"    Colonnes conservées       : {len(colonnes_a_garder)}")

# ========= ÉTAPE 3 : COMPTER AVANT NETTOYAGE =========
print("\n[ÉTAPE 3] Comptage avant nettoyage...")

count_before = pd.read_sql("SELECT COUNT(*) as total FROM histovente;", engine)['total'][0]
print(f"    Lignes avant nettoyage : {count_before:,}")

# ========= ÉTAPE 4 : CRÉER histovente_clean_v1 =========
print("\n[ÉTAPE 4] Création de histovente_clean_v1...")

cols_select = ", ".join([f"`{col}`" for col in colonnes_a_garder])

create_query = f"""
CREATE TABLE histovente_clean_v1 AS
SELECT {cols_select}
FROM histovente
WHERE TypeVente = 'VENTE'
AND Date IS NOT NULL
AND Date >= '2022-01-01'
AND Prix BETWEEN 0 AND 500
AND Quantite BETWEEN 1 AND 100;
"""

print("    Suppression ancienne version si existante...")
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS histovente_clean_v1;"))
    conn.commit()

print("    Création de la table (peut prendre quelques minutes)...")
with engine.connect() as conn:
    conn.execute(text(create_query))
    conn.commit()

print("    ✅ Table histovente_clean_v1 créée !")

# ========= ÉTAPE 5 : VÉRIFICATION =========
print("\n[ÉTAPE 5] Vérification des résultats...")

count_after = pd.read_sql("SELECT COUNT(*) as total FROM histovente_clean_v1;", engine)['total'][0]
lignes_supprimees = count_before - count_after
pct_conserve = round(count_after * 100 / count_before, 2)
pct_supprime = round(lignes_supprimees * 100 / count_before, 2)

print(f"\n    Lignes avant nettoyage  : {count_before:,}")
print(f"    Lignes après nettoyage  : {count_after:,}")
print(f"    Lignes supprimées       : {lignes_supprimees:,} ({pct_supprime}%)")
print(f"    Lignes conservées       : {pct_conserve}%")

# ========= ÉTAPE 6 : STATS PAR FILTRE =========
print("\n[ÉTAPE 6] Détail des filtres appliqués...")

q_type = """
SELECT COUNT(*) as nb 
FROM histovente 
WHERE TypeVente != 'VENTE' OR TypeVente IS NULL;
"""
nb_typevente = pd.read_sql(q_type, engine)['nb'][0]

q_date_null = """
SELECT COUNT(*) as nb 
FROM histovente 
WHERE Date IS NULL;
"""
nb_date_null = pd.read_sql(q_date_null, engine)['nb'][0]

q_date_old = """
SELECT COUNT(*) as nb 
FROM histovente 
WHERE Date IS NOT NULL AND Date < '2022-01-01';
"""
nb_date_old = pd.read_sql(q_date_old, engine)['nb'][0]

q_prix = """
SELECT COUNT(*) as nb 
FROM histovente 
WHERE Prix < 0 OR Prix > 500;
"""
nb_prix = pd.read_sql(q_prix, engine)['nb'][0]

q_qte = """
SELECT COUNT(*) as nb 
FROM histovente 
WHERE Quantite < 1 OR Quantite > 100;
"""
nb_qte = pd.read_sql(q_qte, engine)['nb'][0]

print(f"\n    Filtre TypeVente != VENTE   : {nb_typevente:,} lignes supprimées")
print(f"    Filtre Date IS NULL         : {nb_date_null:,} lignes supprimées")
print(f"    Filtre Date < 2022          : {nb_date_old:,} lignes supprimées")
print(f"    Filtre Prix hors [0-500]    : {nb_prix:,} lignes supprimées")
print(f"    Filtre Quantite hors [1-100]: {nb_qte:,} lignes supprimées")

# ========= ÉTAPE 7 : APERÇU DE LA TABLE PROPRE =========
print("\n[ÉTAPE 7] Aperçu de histovente_clean_v1...")

apercu = pd.read_sql("SELECT * FROM histovente_clean_v1 LIMIT 5;", engine)
print(f"\n    Colonnes disponibles ({len(apercu.columns)}) :")
for col in apercu.columns:
    print(f"    - {col}")

# ========= ÉTAPE 8 : DISTRIBUTION PAR ANNÉE =========
print("\n[ÉTAPE 8] Distribution par année dans la table propre...")

dist_annee = pd.read_sql("""
SELECT 
    YEAR(Date) as annee,
    COUNT(*) as nb_lignes,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pourcentage
FROM histovente_clean_v1
GROUP BY YEAR(Date)
ORDER BY annee;
""", engine)

print(dist_annee.to_string(index=False))

print("\n" + "=" * 70)
print("     DATA CLEANING TERMINÉ ✅")
print("=" * 70)
print(f"\n    Table créée    : histovente_clean_v1")
print(f"    Lignes propres : {count_after:,}")
print(f"    Colonnes       : {len(colonnes_a_garder)}")
print(f"\n    Prochaine étape : Feature Engineering")
print("=" * 70)
