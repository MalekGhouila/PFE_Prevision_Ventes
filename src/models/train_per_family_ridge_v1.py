import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "V2", "hv_france_weekly_features_v2.csv")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "src", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("  PER-FAMILY RIDGE V1 — One model per Famille")
print("=" * 65)

# ===== LOAD =====
df = pd.read_csv(INPUT_PATH, parse_dates=['week_start'])

# ===== FILTER =====
min_avg_sales = 500
high_volume   = df.groupby('Famille')['sum_Quantite'].mean()
high_volume   = high_volume[high_volume >= min_avg_sales].index.tolist()
df = df[df['Famille'].isin(high_volume)]

exclude = ['BIJOUX', 'CHAUSSURE', 'DIVERS', 'CUIR PAM',
           'MAROQUINERIE', 'NAF NAF', 'GROSSES PIÈCES']
df = df[~df['Famille'].isin(exclude)]

FAMILIES = sorted(df['Famille'].unique())
print(f"\n    Families : {len(FAMILIES)}")
print(f"    Kept     : {FAMILIES}")

# ===== FEATURES =====
FEATURES = [
    'year', 'week_of_year', 'month', 'quarter',
    'is_summer', 'is_winter', 'is_soldes', 'is_high_season',
    'lag_1', 'lag_2', 'lag_4', 'lag_52',
    'rolling_mean_4', 'rolling_mean_12'
]
# Note: Famille_encoded removed — each model is already family-specific
TARGET = 'sum_Quantite'

# ===== SPLIT =====
train_df = df[df['week_start'] < '2024-01-01']
test_df  = df[df['week_start'] >= '2024-01-01']

print(f"\n    Train: {train_df['week_start'].min().date()} → {train_df['week_start'].max().date()}")
print(f"    Test : {test_df['week_start'].min().date()} → {test_df['week_start'].max().date()}")

# ===== METRICS =====
def wmape(actual, predicted):
    actual    = np.array(actual)
    predicted = np.array(predicted)
    mask      = actual > 0
    return np.sum(np.abs(actual[mask] - predicted[mask])) / np.sum(actual[mask]) * 100

def evaluate(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return {
        'RMSE' : round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
        'R2'   : round(r2_score(y_true, y_pred), 4)
    }

# ===== TRAIN ONE MODEL PER FAMILY =====
print("\n" + "=" * 65)
print("  PER FAMILLE RESULTS")
print("=" * 65)

all_results  = []
all_preds    = []

for famille in FAMILIES:
    # Filter data for this family only
    train_f = train_df[train_df['Famille'] == famille]
    test_f  = test_df[test_df['Famille'] == famille]

    if len(train_f) < 10 or len(test_f) == 0:
        print(f"    {famille:<22} | ⚠️  Skipped (not enough data)")
        continue

    X_train = train_f[FEATURES].values
    y_train = train_f[TARGET].values
    X_test  = test_f[FEATURES].values
    y_test  = test_f[TARGET].values

    # Scale features — important for Ridge
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train Ridge
    model   = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred  = np.maximum(model.predict(X_test), 0)

    m = evaluate(y_test, y_pred)
    print(f"    {famille:<22} | RMSE={m['RMSE']:.1f} | WMAPE={m['WMAPE']:.1f}% | R²={m['R2']:.3f}")

    all_results.append({'Famille': famille, **m})

    # Save predictions
    preds_f = test_f[['week_start', 'Famille', TARGET]].copy().reset_index(drop=True)
    preds_f['predicted_Quantite'] = y_pred
    all_preds.append(preds_f)

# ===== GLOBAL METRICS =====
all_preds_df = pd.concat(all_preds, ignore_index=True)
y_true_all   = all_preds_df[TARGET].values
y_pred_all   = all_preds_df['predicted_Quantite'].values
global_m     = evaluate(y_true_all, y_pred_all)

print("\n" + "=" * 65)
print(f"  GLOBAL → RMSE={global_m['RMSE']} | WMAPE={global_m['WMAPE']}% | R²={global_m['R2']}")
print("=" * 65)

# ===== COMPARE WITH PREVIOUS GLOBAL RIDGE =====
print(f"\n  Previous (global Ridge) WMAPE : 63.62%")
print(f"  New (per-family Ridge)  WMAPE : {global_m['WMAPE']}%")
diff = round(63.62 - global_m['WMAPE'], 2)
if diff > 0:
    print(f"  Improvement                   : ✅ -{diff}pp")
else:
    print(f"  Change                        : ⚠️  {diff}pp")

# ===== EXPORT =====
results_path = os.path.join(RESULTS_DIR, "per_family_ridge_v1_results.csv")
preds_path   = os.path.join(RESULTS_DIR, "per_family_ridge_v1_predictions.csv")

pd.DataFrame(all_results).to_csv(results_path, index=False)
all_preds_df.to_csv(preds_path, index=False)

print(f"\n  ✅ {results_path}")
print(f"  ✅ {preds_path}")
