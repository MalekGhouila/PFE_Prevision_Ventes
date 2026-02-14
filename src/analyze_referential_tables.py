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

print("=" * 80)
print(" ANALYSE DES COLONNES CLÉS DES TABLES RÉFÉRENTIELS")
print("=" * 80)

# ========= DÉFINITION DES COLONNES PERTINENTES PAR TABLE =========

tables_colonnes = {
    'article': [
        'IDArticle', 'Code', 'Article', 'IDAr_Couleur', 'IDArFamille',
        'Prix', 'PrixAchat', 'IDArSousFamille', 'IDSaison', 'Etat',
        'Reference', 'PrixOutlet'
    ],
    'arfamille': [
        'IDArFamille', 'Famille', 'Code', 'Etat', 'Type', 'CodeDouane'
    ],
    'ar_sfamille': [
        'IDArSousFamille', 'SousFamille', 'Code', 'IDArFamille', 'Etat'
    ],
    'ar_couleur': [
        'IDAr_Couleur', 'Couleur', 'Code', 'Etat', 'Couleur_AN', 'Reference'
    ],
    'magasin': [
        'IDMagasin', 'Magasin', 'Code', 'Etat', 'isBoutique',
        'IDPays', 'IDRegion', 'IDVille', 'IDSecteur', 'IDCategorie'
    ],
    'saison': [
        'IDSaison', 'Saison', 'Code', 'Etat', 'DateDebut', 'DateFin'
    ]
}

# ========= ANALYSE COLONNE PAR COLONNE =========

resultats = []

for table_name, colonnes in tables_colonnes.items():
    print(f"\n[TABLE : {table_name}]")
    print(f"    Analyse de {len(colonnes)} colonnes pertinentes...")

    # Compter le total de lignes de la table
    query_total = f"SELECT COUNT(*) as total FROM {table_name};"
    total_lignes = pd.read_sql(query_total, engine)['total'][0]

    for idx, col in enumerate(colonnes, 1):
        print(f"    [{idx}/{len(colonnes)}] {col}...", end=" ")

        try:
            # Requête universelle pour stats de base
            query_stats = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(`{col}`) as valeurs_remplies,
                COUNT(DISTINCT `{col}`) as valeurs_distinctes
            FROM {table_name};
            """
            stats = pd.read_sql(query_stats, engine).iloc[0]

            valeurs_remplies = int(stats['valeurs_remplies'])
            valeurs_nulles = total_lignes - valeurs_remplies
            pct_null = round(valeurs_nulles * 100.0 / total_lignes, 2) if total_lignes > 0 else 0
            valeurs_distinctes = int(stats['valeurs_distinctes'])

            # Top 10 valeurs pour colonnes catégorielles
            query_top = f"""
            SELECT `{col}`, COUNT(*) as nb
            FROM {table_name}
            WHERE `{col}` IS NOT NULL
            GROUP BY `{col}`
            ORDER BY nb DESC
            LIMIT 10;
            """
            top_valeurs = pd.read_sql(query_top, engine)

            if len(top_valeurs) > 0:
                top_str = ", ".join([f"{row[col]} ({row['nb']:,})" for _, row in top_valeurs.iterrows()])
            else:
                top_str = "Aucune valeur"

            # Interprétation auto
            if col.startswith('ID'):
                if valeurs_distinctes == valeurs_remplies:
                    interpretation = "Clé primaire ou identifiant unique"
                else:
                    interpretation = "Clé étrangère"
            elif pct_null > 50:
                interpretation = "Colonne peu fiable (>50% NULL)"
            elif valeurs_distinctes < 10:
                interpretation = f"Faible cardinalité ({valeurs_distinctes} valeurs)"
            elif valeurs_distinctes == total_lignes:
                interpretation = "Identifiant unique ou label unique"
            else:
                interpretation = f"Variable catégorique ({valeurs_distinctes} valeurs distinctes)"

            resultats.append({
                'Table': table_name,
                'Colonne': col,
                'Total_Lignes': total_lignes,
                'Valeurs_Remplies': valeurs_remplies,
                'Valeurs_Nulles': valeurs_nulles,
                'Pct_NULL': pct_null,
                'Valeurs_Distinctes': valeurs_distinctes,
                'Top_10_Valeurs': top_str,
                'Interpretation_Auto': interpretation
            })

            print("✓")

        except Exception as e:
            print(f"✗ ERREUR : {e}")
            resultats.append({
                'Table': table_name,
                'Colonne': col,
                'Erreur': str(e)
            })

# ========= SAUVEGARDER LES RÉSULTATS =========
print("\n" + "=" * 80)
print(" SAUVEGARDE DES RÉSULTATS")
print("=" * 80)

df_resultats = pd.DataFrame(resultats)
df_resultats.to_excel("datastories_referentiels.xlsx", index=False, sheet_name="Colonnes Référentiels")

print(f"\n✅ Fichier créé : datastories_referentiels.xlsx")

# ========= RÉSUMÉ =========
print("\n" + "=" * 80)
print(" RÉSUMÉ")
print("=" * 80)

for table_name in tables_colonnes.keys():
    nb_cols = len(tables_colonnes[table_name])
    total = df_resultats[df_resultats['Table'] == table_name]['Total_Lignes'].values[0]
    print(f"\n📊 {table_name} : {nb_cols} colonnes analysées, {total:,} lignes")

print("\n" + "=" * 80)
print(" ANALYSE TERMINÉE")
print("=" * 80)
print("\n📁 Fichiers générés :")
print("   1. datastories_histovente_62colonnes.xlsx")
print("   2. datastories_referentiels.xlsx")
print("\n📧 Envoie ces 2 fichiers Excel pour générer le document Word final\n")
