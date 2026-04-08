import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  XGBOOST V6 — Weekly, One model per Famille")
print("=" * 60)

df = pd.read_csv("hv_france_weekly_features.csv", parse_dates=['week_start'])
print(f"\n[1] Rows: {len(df):,} | Families: {df['Famille'].nunique()}")
print(f"    Date range: {df['week_start'].min().date()} → {df['week_start'].max().date()}")

FEATURES = [
    'year', 'week_of_year', 'month', 'quarter',
    'is_summer', 'is_winter', 'is_soldes',
    'Famille_encoded',
    'lag_1', 'lag_2', 'lag_4',
    'rolling_mean_4', 'rolling_mean_12'
]
TARGET = 'sum_Quantite'

results   = []
all_preds = []

familles = sorted(df['Famille'].dropna().unique())
print(f"\n[2] Training {len(familles)} models...")
print(f"    Train: Jul 2022 → Dec 2024")
print(f"    Test : Jan 2025 → May 2025\n")

for famille in familles:
    df_f = df[df['Famille'] == famille].copy().sort_values('week_start')

    train = df_f[df_f['week_start'] < '2025-01-01']
    test  = df_f[df_f['week_start'] >= '2025-01-01']

    if len(train) < 10 or len(test) < 3:
        print(f"    ⚠️  {famille} skipped (train={len(train)}, test={len(test)})")
        continue

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
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

    mask     = y_test >= 5
    mape     = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else None
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
    r2       = r2_score(y_test, y_pred)
    mape_str = f"{mape:.1f}%" if mape is not None else "N/A"

    print(f"    ✅ {famille:<22} | train={len(train)}w test={len(test)}w | RMSE={rmse:.1f} | MAPE={mape_str} | R²={r2:.3f}")

    results.append({'Famille': famille, 'RMSE': rmse, 'MAPE': mape, 'R2': r2,
                    'train_weeks': len(train), 'test_weeks': len(test)})

    test_copy = test.copy()
    test_copy['predicted_Quantite'] = y_pred
    all_preds.append(test_copy[['week_start', 'Famille', 'sum_Quantite', 'predicted_Quantite']])

# ===== RESULTS =====
if len(results) == 0:
    print("\n❌ No models trained!")
else:
    results_df   = pd.DataFrame(results)
    all_preds_df = pd.concat(all_preds, ignore_index=True)

    print("\n" + "=" * 60)
    print("  OVERALL RESULTS")
    print("=" * 60)
    print(f"  Models trained : {len(results_df)}")
    print(f"  Avg MAPE       : {results_df['MAPE'].mean():.2f}%")
    print(f"  Avg R²         : {results_df['R2'].mean():.4f}")
    print(f"  Avg RMSE       : {results_df['RMSE'].mean():.2f}")
    print("=" * 60)

    results_df.to_csv("v6_results_per_famille.csv", index=False)
    all_preds_df.to_csv("v6_predictions.csv", index=False)
    print("\n  ✅ v6_results_per_famille.csv")
    print("  ✅ v6_predictions.csv")