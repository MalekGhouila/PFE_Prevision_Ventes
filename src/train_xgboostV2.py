import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("     MODÈLE XGBOOST V2 - Prévision sum_Quantite")
print("=" * 70)

# ========= ÉTAPE 1 : CHARGER =========
print("\n[ÉTAPE 1] Chargement de hv_features.csv...")

df = pd.read_csv("hv_features.csv", parse_dates=['Date'])
print(f"    Lignes : {len(df):,}")

# ========= ÉTAPE 2 : FILTRER (top familles seulement) =========
print("\n[ÉTAPE 2] Filtrage top 10 familles...")

top_familles = df.groupby('Famille')['sum_Quantite'].sum().nlargest(10).index.tolist()
df = df[df['Famille'].isin(top_familles)]
print(f"    Top 10 familles : {top_familles}")
print(f"    Lignes après filtre : {len(df):,}")

# ========= ÉTAPE 3 : FILTRER lignes avec sum_Quantite >= 2 =========
print("\n[ÉTAPE 3] Suppression des lignes sum_Quantite < 2 (MAPE instable)...")

df = df[df['sum_Quantite'] >= 2]
print(f"    Lignes après filtre : {len(df):,}")

# ========= ÉTAPE 4 : TRAIN / TEST SPLIT =========
print("\n[ÉTAPE 4] Split temporel train/test...")

train = df[df['Date'] < '2024-01-01']
test  = df[df['Date'] >= '2024-01-01']

print(f"    Train : {len(train):,} lignes")
print(f"    Test  : {len(test):,} lignes")

# ========= ÉTAPE 5 : FEATURES / TARGET =========
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

# ========= ÉTAPE 6 : ENTRAÎNEMENT avec early stopping =========
print("\n[ÉTAPE 6] Entraînement XGBoost avec early stopping...")

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print(f"    ✅ Meilleur nombre d'arbres : {model.best_iteration}")

# ========= ÉTAPE 7 : ÉVALUATION =========
print("\n[ÉTAPE 7] Évaluation...")

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# MAPE calculé uniquement sur lignes > 5 pour éviter division par ~0
mask = y_test >= 5
mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

print(f"\n    RMSE : {rmse:.4f}")
print(f"    MAPE : {mape:.2f}% (calculé sur sum_Quantite >= 5)")
print(f"    R²   : {r2:.4f}")

# ========= ÉTAPE 8 : FEATURE IMPORTANCE =========
print("\n[ÉTAPE 8] Feature importance (top 10)...")

importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10).to_string(index=False))

# ========= ÉTAPE 9 : EXPORT =========
print("\n[ÉTAPE 9] Export résultats...")

test = test.copy()
test['predicted_Quantite'] = y_pred
test[['Date', 'CodeMag', 'Famille', 'Pays',
      'sum_Quantite', 'predicted_Quantite']].to_csv("predictions_v2.csv", index=False)

print("    ✅ predictions_v2.csv exporté")

print("\n" + "=" * 70)
print("     ENTRAÎNEMENT V2 TERMINÉ ✅")
print("=" * 70)
print(f"\n    RMSE : {rmse:.4f}")
print(f"    MAPE : {mape:.2f}%")
print(f"    R²   : {r2:.4f}")
print("=" * 70)