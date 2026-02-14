import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
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
print(" ANALYSE AUTOMATIQUE DES 62 COLONNES DE HISTOVENTE")
print("=" * 80)

# ========= ÉTAPE 1 : Récupérer la liste des colonnes =========
print("\n[1] Récupération de la structure de la table...")

query_structure = """
                  SELECT COLUMN_NAME, \
                         DATA_TYPE, \
                         IS_NULLABLE, \
                         COLUMN_KEY, \
                         ORDINAL_POSITION
                  FROM INFORMATION_SCHEMA.COLUMNS
                  WHERE TABLE_SCHEMA = 'erp_sales'
                    AND TABLE_NAME = 'histovente'
                  ORDER BY ORDINAL_POSITION; \
                  """

df_structure = pd.read_sql(query_structure, engine)
colonnes = df_structure['COLUMN_NAME'].tolist()

print(f"    → {len(colonnes)} colonnes détectées")

# ========= ÉTAPE 2 : Compter le total de lignes =========
print("\n[2] Comptage du total de lignes...")

query_total = "SELECT COUNT(*) as total FROM histovente;"
total_lignes = pd.read_sql(query_total, engine)['total'][0]

print(f"    → Total : {total_lignes:,} lignes")

# ========= ÉTAPE 3 : Analyser chaque colonne =========
print("\n[3] Analyse colonne par colonne...")
print("    (Cela peut prendre 10-20 minutes sur 16M lignes)")

resultats = []

for idx, col in enumerate(colonnes, 1):
    print(f"\n    [{idx}/{len(colonnes)}] Analyse de '{col}'...", end=" ")

    try:
        # Infos de base
        type_sql = df_structure[df_structure['COLUMN_NAME'] == col]['DATA_TYPE'].values[0]
        is_nullable = df_structure[df_structure['COLUMN_NAME'] == col]['IS_NULLABLE'].values[0]
        column_key = df_structure[df_structure['COLUMN_NAME'] == col]['COLUMN_KEY'].values[0]

        # Déterminer le type logique
        if col.startswith('ID') or 'ID' in col.upper():
            type_logique = "Identifiant numérique"
        elif type_sql in ['date', 'datetime', 'timestamp']:
            type_logique = "Temporelle"
        elif type_sql in ['int', 'bigint', 'smallint', 'tinyint', 'mediumint']:
            type_logique = "Numérique discrète"
        elif type_sql in ['decimal', 'float', 'double']:
            type_logique = "Numérique continue"
        elif type_sql in ['varchar', 'char', 'text', 'longtext']:
            type_logique = "Catégorique nominale"
        else:
            type_logique = "Autre"

        # Requête adaptée selon le type
        if type_sql in ['int', 'bigint', 'smallint', 'tinyint', 'mediumint', 'decimal', 'float', 'double']:
            # Colonne numérique
            query_stats = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(`{col}`) as valeurs_remplies,
                COUNT(DISTINCT `{col}`) as valeurs_distinctes,
                MIN(`{col}`) as min_val,
                MAX(`{col}`) as max_val,
                AVG(`{col}`) as moyenne,
                STD(`{col}`) as ecart_type
            FROM histovente;
            """
            stats = pd.read_sql(query_stats, engine).iloc[0]

            valeurs_remplies = int(stats['valeurs_remplies'])
            valeurs_nulles = total_lignes - valeurs_remplies
            pct_null = round(valeurs_nulles * 100.0 / total_lignes, 2)
            valeurs_distinctes = int(stats['valeurs_distinctes'])
            min_val = stats['min_val']
            max_val = stats['max_val']
            moyenne = round(stats['moyenne'], 2) if pd.notna(stats['moyenne']) else None
            ecart_type = round(stats['ecart_type'], 2) if pd.notna(stats['ecart_type']) else None

            # Top valeurs
            query_top = f"""
            SELECT `{col}`, COUNT(*) as nb
            FROM histovente
            WHERE `{col}` IS NOT NULL
            GROUP BY `{col}`
            ORDER BY nb DESC
            LIMIT 5;
            """
            top_valeurs = pd.read_sql(query_top, engine)
            top_str = ", ".join([f"{row[col]} ({row['nb']:,})" for _, row in top_valeurs.iterrows()])

            mediane = None  # Trop coûteux en calcul, on skip

            # Interprétation auto
            if column_key == 'PRI':
                interpretation = "Clé primaire"
            elif pct_null > 50:
                interpretation = "Colonne peu fiable (>50% NULL)"
            elif valeurs_distinctes == valeurs_remplies and valeurs_remplies > 1000:
                interpretation = "Probablement un identifiant unique"
            elif max_val and moyenne and max_val > 1000 * moyenne:
                interpretation = "Outliers extrêmes détectés"
            elif valeurs_distinctes < 10:
                interpretation = "Faible cardinalité (booléen/énuméré)"
            else:
                interpretation = "Variable numérique standard"

        elif type_sql in ['date', 'datetime', 'timestamp']:
            # Colonne temporelle
            query_stats = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(`{col}`) as valeurs_remplies,
                COUNT(DISTINCT `{col}`) as valeurs_distinctes,
                MIN(`{col}`) as min_val,
                MAX(`{col}`) as max_val
            FROM histovente;
            """
            stats = pd.read_sql(query_stats, engine).iloc[0]

            valeurs_remplies = int(stats['valeurs_remplies'])
            valeurs_nulles = total_lignes - valeurs_remplies
            pct_null = round(valeurs_nulles * 100.0 / total_lignes, 2)
            valeurs_distinctes = int(stats['valeurs_distinctes'])
            min_val = stats['min_val']
            max_val = stats['max_val']
            moyenne = None
            ecart_type = None
            mediane = None
            top_str = f"Période : {min_val} → {max_val}"

            if pct_null > 50:
                interpretation = "Colonne peu fiable (>50% NULL)"
            elif valeurs_distinctes < 10:
                interpretation = "Peu de dates distinctes, vérifier cohérence"
            else:
                interpretation = f"Couvre {valeurs_distinctes} jours distincts"

        else:
            # Colonne catégorique/texte
            query_stats = f"""
            SELECT 
                COUNT(*) as total,
                COUNT(`{col}`) as valeurs_remplies,
                COUNT(DISTINCT `{col}`) as valeurs_distinctes
            FROM histovente;
            """
            stats = pd.read_sql(query_stats, engine).iloc[0]

            valeurs_remplies = int(stats['valeurs_remplies'])
            valeurs_nulles = total_lignes - valeurs_remplies
            pct_null = round(valeurs_nulles * 100.0 / total_lignes, 2)
            valeurs_distinctes = int(stats['valeurs_distinctes'])
            min_val = None
            max_val = None
            moyenne = None
            ecart_type = None
            mediane = None

            # Top valeurs
            query_top = f"""
            SELECT `{col}`, COUNT(*) as nb
            FROM histovente
            WHERE `{col}` IS NOT NULL
            GROUP BY `{col}`
            ORDER BY nb DESC
            LIMIT 5;
            """
            top_valeurs = pd.read_sql(query_top, engine)
            top_str = ", ".join(
                [f"{row[col]} ({round(row['nb'] * 100 / total_lignes, 1)}%)" for _, row in top_valeurs.iterrows()])

            if pct_null > 50:
                interpretation = "Colonne peu fiable (>50% NULL)"
            elif valeurs_distinctes == valeurs_remplies and valeurs_remplies > 1000:
                interpretation = "Probablement un identifiant unique"
            elif valeurs_distinctes < 10:
                interpretation = f"Faible cardinalité ({valeurs_distinctes} valeurs)"
            elif valeurs_distinctes > 1000:
                interpretation = f"Haute cardinalité ({valeurs_distinctes:,} valeurs distinctes)"
            else:
                interpretation = "Variable catégorique standard"

        # Ajouter aux résultats
        resultats.append({
            'Colonne': col,
            'Type_SQL': type_sql,
            'Type_Logique': type_logique,
            'Clé': column_key if column_key else '',
            'Total_Lignes': total_lignes,
            'Valeurs_Remplies': valeurs_remplies,
            'Valeurs_Nulles': valeurs_nulles,
            'Pct_NULL': pct_null,
            'Valeurs_Distinctes': valeurs_distinctes,
            'Min': min_val,
            'Max': max_val,
            'Moyenne': moyenne,
            'Ecart_Type': ecart_type,
            'Mediane': mediane,
            'Top_5_Valeurs': top_str,
            'Interpretation_Auto': interpretation
        })

        print("✓")

    except Exception as e:
        print(f"✗ ERREUR : {e}")
        resultats.append({
            'Colonne': col,
            'Type_SQL': type_sql,
            'Type_Logique': type_logique,
            'Erreur': str(e)
        })

# ========= ÉTAPE 4 : Sauvegarder les résultats =========
print("\n[4] Sauvegarde des résultats...")

df_resultats = pd.DataFrame(resultats)
df_resultats.to_excel("datastories_histovente_62colonnes.xlsx", index=False, sheet_name="Analyse Colonnes")

print(f"    → Fichier créé : datastories_histovente_62colonnes.xlsx")

# ========= ÉTAPE 5 : Afficher un résumé =========
print("\n" + "=" * 80)
print(" RÉSUMÉ DE L'ANALYSE")
print("=" * 80)

print(f"\n✅ {len(colonnes)} colonnes analysées")
print(f"✅ {total_lignes:,} lignes dans histovente")

print("\n📊 Répartition par type logique :")
print(df_resultats['Type_Logique'].value_counts())

print("\n⚠️  Colonnes avec >10% de valeurs nulles :")
nulls_importants = df_resultats[df_resultats['Pct_NULL'] > 10].sort_values('Pct_NULL', ascending=False)
if len(nulls_importants) > 0:
    for _, row in nulls_importants.iterrows():
        print(f"    - {row['Colonne']} : {row['Pct_NULL']}% NULL")
else:
    print("    Aucune")

print("\n" + "=" * 80)
print(" ANALYSE TERMINÉE")
print("=" * 80)
print("\n📁 Fichier : datastories_histovente_62colonnes.xlsx")
print("📧 Envoie ce fichier à ton assistant pour générer le document Word final\n")
