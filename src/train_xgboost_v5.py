import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  XGBOOST V5 — One model per Famille (France Total)")
print("=" * 60)

# ===== LOAD =====
df = pd.read_csv("hv_france_features.csv", parse_dates=['Date'])
print(f"\n[1] Rows: {len(df):,} | Families: {df['Famille'].nunique()}")

FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
    'Famille_encoded',
    'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_mean_30'
]
TARGET = 'sum_Quantite'

# ===== TRAIN ONE MODEL PER FAMILLE =====
results = []
all_preds = []

familles = sorted(df['Famille'].dropna().unique())
print(f"\n[2] Training {len(familles)} models...\n")

for famille in familles:
    df_f = df[df['Famille'] == famille].copy().sort_values('Date')

    train = df_f[df_f['Date'] < '2024-01-01']
    test  = df_f[df_f['Date'] >= '2024-01-01']

    if len(train) < 50 or len(test) < 10:
        print(f"    ⚠️  {famille} skipped (not enough data)")
        continue

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = np.maximum(model.predict(X_test), 0)

    # MAPE only on rows >= 5
    mask = y_test >= 5
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else None
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    mape_str = f"{mape:.1f}%" if mape is not None else "N/A"
    print(f"    ✅ {famille:<20} | RMSE={rmse:.1f} | MAPE={mape_str} | R²={r2:.3f}")
    results.append({'Famille': famille, 'RMSE': rmse, 'MAPE': mape, 'R2': r2,
                    'train_rows': len(train), 'test_rows': len(test)})

    test_copy = test.copy()
    test_copy['predicted_Quantite'] = y_pred
    all_preds.append(test_copy[['Date', 'Famille', 'sum_Quantite', 'predicted_Quantite']])

# ===== RESULTS =====
results_df = pd.DataFrame(results)
all_preds_df = pd.concat(all_preds, ignore_index=True)

print("\n" + "=" * 60)
print("  OVERALL RESULTS")
print("=" * 60)
print(f"  Avg MAPE : {results_df['MAPE'].mean():.2f}%")
print(f"  Avg R²   : {results_df['R2'].mean():.4f}")
print(f"  Avg RMSE : {results_df['RMSE'].mean():.2f}")
print("=" * 60)

results_df.to_csv("v5_results_per_famille.csv", index=False)
all_preds_df.to_csv("v5_predictions.csv", index=False)
print("\n  ✅ Exported: v5_results_per_famille.csv")
print("  ✅ Exported: v5_predictions.csv")