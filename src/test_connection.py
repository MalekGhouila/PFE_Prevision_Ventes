import mysql.connector

print(" Test de connexion à MySQL...")
print("=" * 50)

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "erp_sales",
}

try:
    print("\n Tentative de connexion...")
    conn = mysql.connector.connect(**DB_CONFIG)

    if conn.is_connected():
        print(" Connexion réussie à MySQL !")

        cursor = conn.cursor()

        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f" Version MySQL : {version}")

        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f" Base sélectionnée : {db_name}")

        cursor.execute("SHOW TABLES LIKE 'histovente'")
        table = cursor.fetchone()
        if table:
            print(" Table 'histovente' trouvée.")
            cursor.execute("SELECT COUNT(*) FROM histovente")
            total = cursor.fetchone()[0]
            print(f" Nombre de lignes dans histovente : {total:,}")
        else:
            print(" Table 'histovente' non trouvée dans cette base.")

        cursor.close()
        conn.close()
        print("\n Test terminé avec succès.")
    else:
        print(" Connexion non établie (is_connected() = False).")

except mysql.connector.Error as err:
    print(f"\n ERREUR MySQL : {err}")
except Exception as e:
    print(f"\n ERREUR Python : {e}")
