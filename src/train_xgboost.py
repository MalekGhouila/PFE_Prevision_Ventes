import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("     MODÈLE XGBOOST - Prévision sum_Quantite")
print("=" * 70)

# ========= ÉTAPE 1 : CHARGER LES FEATURES =========
print("\n[ÉTAPE 1] Chargement de hv_features.csv...")

df = pd.read_csv("hv_features.csv", parse_dates=['Date'])
print(f"    Lignes : {len(df):,}")
print(f"    Période : {df['Date'].min().date()} → {df['Date'].max().date()}")

# ========= ÉTAPE 2 : TRAIN / TEST SPLIT =========
print("\n[ÉTAPE 2] Split temporel train/test...")

train = df[df['Date'] < '2024-01-01']
test  = df[df['Date'] >= '2024-01-01']

print(f"    Train : {len(train):,} lignes ({train['Date'].min().date()} → {train['Date'].max().date()})")
print(f"    Test  : {len(test):,}  lignes ({test['Date'].min().date()} → {test['Date'].max().date()})")

# ========= ÉTAPE 3 : FEATURES / TARGET =========
print("\n[ÉTAPE 3] Définition features et target...")

FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
    'Pays_encoded', 'Famille_encoded', 'CodeMag_encoded',
    'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_mean_30'
]
TARGET = 'sum_Quantite'

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"    Features utilisées : {len(FEATURES)}")
print(f"    Target             : {TARGET}")

# ========= ÉTAPE 4 : ENTRAÎNEMENT =========
print("\n[ÉTAPE 4] Entraînement XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print("    ✅ Entraînement terminé !")

# ========= ÉTAPE 5 : ÉVALUATION =========
print("\n[ÉTAPE 5] Évaluation sur le test set...")

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # pas de quantité négative

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2   = r2_score(y_test, y_pred)

print(f"\n    RMSE : {rmse:.4f}")
print(f"    MAPE : {mape:.2f}%")
print(f"    R²   : {r2:.4f}")

# ========= ÉTAPE 6 : FEATURE IMPORTANCE =========
print("\n[ÉTAPE 6] Feature importance (top 10)...")

importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10).to_string(index=False))

# ========= ÉTAPE 7 : EXPORT RÉSULTATS =========
print("\n[ÉTAPE 7] Export des résultats...")

test = test.copy()
test['predicted_Quantite'] = y_pred
test[['Date', 'CodeMag', 'Famille', 'Pays',
      'sum_Quantite', 'predicted_Quantite']].to_csv("predictions.csv", index=False)

print("    ✅ Fichier exporté : predictions.csv")

print("\n" + "=" * 70)
print("     ENTRAÎNEMENT TERMINÉ ✅")
print("=" * 70)
print(f"\n    RMSE : {rmse:.4f}")
print(f"    MAPE : {mape:.2f}%")
print(f"    R²   : {r2:.4f}")
print("=" * 70)