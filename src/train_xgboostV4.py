import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("     MODÈLE XGBOOST V4 - France only, un modèle par famille")
print("=" * 70)

# ========= CHARGEMENT =========
print("\n[CHARGEMENT] hv_features.csv...")
df = pd.read_csv("hv_features.csv", parse_dates=['Date'])

# France only + filtre quantité
df = df[(df['Pays'] == 'FRANCE') & (df['sum_Quantite'] >= 2)]
print(f"    Lignes après filtre France + qty>=2 : {len(df):,}")

# Top 10 familles par volume
top_familles = df.groupby('Famille')['sum_Quantite'].sum().nlargest(10).index.tolist()
print(f"    Top 10 familles : {top_familles}")

# Pays_encoded retiré car tout est FRANCE (valeur constante = inutile)
FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
    'Famille_encoded', 'CodeMag_encoded',
    'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_mean_30'
]
TARGET = 'sum_Quantite'

results  = []
all_preds = []

print("\n[ENTRAÎNEMENT] Un modèle par famille...\n")

for famille in top_familles:
    df_f = df[df['Famille'] == famille].copy()

    train = df_f[df_f['Date'] < '2024-01-01']
    test  = df_f[df_f['Date'] >= '2024-01-01']

    if len(train) < 100 or len(test) < 10:
        print(f"    ⚠️  {famille:<25} ignorée (pas assez de données)")
        continue

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = np.maximum(model.predict(X_test), 0)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mask   = y_test >= 5
    mape   = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else 999

    results.append({
        'Famille'   : famille,
        'train_size': len(train),
        'test_size' : len(test),
        'best_iter' : model.best_iteration,
        'RMSE'      : round(rmse, 2),
        'MAPE'      : round(mape, 2),
        'R2'        : round(r2, 4)
    })

    test = test.copy()
    test['predicted_Quantite'] = y_pred
    all_preds.append(test[['Date', 'CodeMag', 'Famille', 'Pays',
                            'sum_Quantite', 'predicted_Quantite']])

    print(f"    ✅ {famille:<25} RMSE={rmse:.2f}  MAPE={mape:.1f}%  R²={r2:.4f}  best_iter={model.best_iteration}")

# ========= RÉSUMÉ =========
df_results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("     RÉSUMÉ PAR FAMILLE")
print("=" * 70)
print(df_results.to_string(index=False))

print(f"\n    MAPE moyen  : {df_results['MAPE'].mean():.2f}%")
print(f"    R² moyen    : {df_results['R2'].mean():.4f}")
print(f"    RMSE moyen  : {df_results['RMSE'].mean():.2f}")

# ========= EXPORT =========
pd.concat(all_preds).to_csv("predictions_v4.csv", index=False)
df_results.to_csv("model_results_v4.csv", index=False)

print("\n    ✅ predictions_v4.csv exporté")
print("    ✅ model_results_v4.csv exporté")
print("=" * 70)