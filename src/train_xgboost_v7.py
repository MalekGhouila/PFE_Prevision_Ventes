import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  XGBOOST V7 — One Global Model, Core Clothing Only")
print("=" * 60)

df = pd.read_csv("hv_france_weekly_features.csv", parse_dates=['week_start'])

# Keep only high-volume families
min_avg_sales = 500
high_volume = df.groupby('Famille')['sum_Quantite'].mean()
high_volume = high_volume[high_volume >= min_avg_sales].index.tolist()
df = df[df['Famille'].isin(high_volume)]

# Remove accessory/non-core families
exclude = ['BIJOUX', 'CHAUSSURE', 'DIVERS', 'CUIR PAM',
           'MAROQUINERIE', 'NAF NAF', 'GROSSES PIÈCES']
df = df[~df['Famille'].isin(exclude)]

print(f"    Core clothing families: {df['Famille'].nunique()}")
print(f"    Families: {sorted(df['Famille'].unique())}")

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

# ===== SPLIT =====
train = df[df['week_start'] < '2025-01-01']
test  = df[df['week_start'] >= '2025-01-01']

print(f"\n[2] Train/Test split...")
print(f"    Train: {len(train):,} rows ({train['week_start'].min().date()} → {train['week_start'].max().date()})")
print(f"    Test : {len(test):,} rows ({test['week_start'].min().date()} → {test['week_start'].max().date()})")

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

# ===== TRAIN =====
print(f"\n[3] Training on {len(train):,} rows...")

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=100)

print("    ✅ Training done!")

# ===== WMAPE FUNCTION =====
def wmape(actual, predicted):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    return np.sum(np.abs(actual - predicted)) / np.sum(actual) * 100

# ===== EVALUATE GLOBAL =====
y_pred = np.maximum(model.predict(X_test), 0)
wm     = wmape(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print(f"\n[4] Global results:")
print(f"    RMSE  : {rmse:.2f}")
print(f"    WMAPE : {wm:.2f}%")
print(f"    R²    : {r2:.4f}")

# ===== EVALUATE PER FAMILLE =====
print(f"\n[5] Results per Famille:")
test_copy = test.copy()
test_copy['predicted_Quantite'] = y_pred

results = []
for famille in sorted(test_copy['Famille'].unique()):
    df_f   = test_copy[test_copy['Famille'] == famille]
    yt     = df_f[TARGET].values
    yp     = df_f['predicted_Quantite'].values
    wm_f   = wmape(yt, yp)
    rmse_f = np.sqrt(mean_squared_error(yt, yp))
    r2_f   = r2_score(yt, yp)
    print(f"    {famille:<22} | RMSE={rmse_f:.1f} | WMAPE={wm_f:.1f}% | R²={r2_f:.3f}")
    results.append({'Famille': famille, 'RMSE': rmse_f, 'WMAPE': wm_f, 'R2': r2_f})

# ===== FEATURE IMPORTANCE =====
print(f"\n[6] Top 10 feature importance:")
importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# ===== EXPORT =====
results_df = pd.DataFrame(results)
results_df.to_csv("v7_results_per_famille.csv", index=False)
test_copy[['week_start', 'Famille', 'sum_Quantite',
           'predicted_Quantite']].to_csv("v7_predictions.csv", index=False)

print(f"\n" + "=" * 60)
print(f"  FINAL RESULTS")
print(f"=" * 60)
print(f"  RMSE  : {rmse:.2f}")
print(f"  WMAPE : {wm:.2f}%")
print(f"  R²    : {r2:.4f}")
print(f"=" * 60)
print(f"  ✅ v7_results_per_famille.csv")
print(f"  ✅ v7_predictions.csv")