import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "V2", "hv_france_weekly_features_v2.csv")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "src", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("  MODEL COMPARISON V1 — XGB / LGBM / RF / GBR / Ridge / Prophet / Naive")
print("=" * 70)

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
    'Famille_encoded',
    'lag_1', 'lag_2', 'lag_4', 'lag_52',
    'rolling_mean_4', 'rolling_mean_12'
]
TARGET = 'sum_Quantite'

# ===== SPLIT =====
train = df[df['week_start'] < '2024-01-01']
test  = df[df['week_start'] >= '2024-01-01']

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\n    Train: {train['week_start'].min().date()} → {train['week_start'].max().date()} ({len(train):,} rows)")
print(f"    Test : {test['week_start'].min().date()} → {test['week_start'].max().date()} ({len(test):,} rows)")

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

# ===== MODELS =====
ML_MODELS = {
    'XGBoost': XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse'
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5,
        num_leaves=31, subsample=0.8, random_state=42,
        n_jobs=-1, verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
    ),
    'Ridge': Ridge(alpha=1.0),
}

# ===== TRAIN ML MODELS =====
global_results = []
all_preds      = test[['week_start', 'Famille', TARGET]].copy().reset_index(drop=True)

print("\n" + "=" * 70)
print("  GLOBAL RESULTS")
print("=" * 70)

for name, model in ML_MODELS.items():
    print(f"\n[{name}] Training...")

    if name == 'XGBoost':
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)], verbose=False)
    elif name == 'LightGBM':
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[])
    else:
        model.fit(X_train, y_train)

    y_pred = np.maximum(model.predict(X_test), 0)
    metrics = evaluate(y_test, y_pred)
    all_preds[f'pred_{name}'] = y_pred

    print(f"    RMSE={metrics['RMSE']} | WMAPE={metrics['WMAPE']}% | R²={metrics['R2']}")
    global_results.append({'Model': name, **metrics})

# ===== NAIVE BASELINE (lag_52) =====
print(f"\n[Naive lag_52] No training needed...")
naive_pred    = np.maximum(test['lag_52'].fillna(y_test.mean()).values, 0)
naive_metrics = evaluate(y_test, naive_pred)
all_preds['pred_Naive'] = naive_pred
print(f"    RMSE={naive_metrics['RMSE']} | WMAPE={naive_metrics['WMAPE']}% | R²={naive_metrics['R2']}")
global_results.append({'Model': 'Naive (lag_52)', **naive_metrics})

# ===== PROPHET (per family) =====
print(f"\n[Prophet] Training per family...")
prophet_preds = []

for famille in FAMILIES:
    df_fam         = df[df['Famille'] == famille][['week_start', TARGET]].copy()
    df_fam.columns = ['ds', 'y']
    df_fam['y']    = df_fam['y'].clip(lower=0)

    df_train_p = df_fam[df_fam['ds'] < '2024-01-01']
    df_test_p  = df_fam[df_fam['ds'] >= '2024-01-01']

    if len(df_train_p) < 10 or len(df_test_p) == 0:
        continue

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1
    )
    m.fit(df_train_p)

    forecast = m.predict(df_test_p[['ds']])
    yhat     = np.maximum(forecast['yhat'].values, 0)

    prophet_preds.append({
        'y_true': df_test_p['y'].values,
        'y_pred': yhat
    })

prophet_true    = np.concatenate([p['y_true'] for p in prophet_preds])
prophet_pred    = np.concatenate([p['y_pred'] for p in prophet_preds])
prophet_metrics = evaluate(prophet_true, prophet_pred)
print(f"    RMSE={prophet_metrics['RMSE']} | WMAPE={prophet_metrics['WMAPE']}% | R²={prophet_metrics['R2']}")
global_results.append({'Model': 'Prophet', **prophet_metrics})

# ===== SUMMARY TABLE =====
results_df = pd.DataFrame(global_results).sort_values('WMAPE').reset_index(drop=True)

print("\n" + "=" * 70)
print("  FINAL RANKING — sorted by WMAPE ↑")
print("=" * 70)
print(results_df.to_string(index=False))

# ===== PER FAMILLE (best ML model) =====
# ← FIXED: clean filtering without index misalignment
exclude_from_best = ['Prophet', 'Naive (lag_52)']
ml_only_df        = results_df[~results_df['Model'].isin(exclude_from_best)].reset_index(drop=True)
best_ml_model_name = ml_only_df.loc[ml_only_df['WMAPE'].idxmin(), 'Model']

print(f"\n\n  Best ML model: {best_ml_model_name}")
print("=" * 70)
print(f"  PER FAMILLE — {best_ml_model_name}")
print("=" * 70)

for famille in FAMILIES:
    df_f   = all_preds[all_preds['Famille'] == famille]
    yt, yp = df_f[TARGET].values, df_f[f'pred_{best_ml_model_name}'].values
    m      = evaluate(yt, yp)
    print(f"    {famille:<22} | RMSE={m['RMSE']:.1f} | WMAPE={m['WMAPE']:.1f}% | R²={m['R2']:.3f}")

# ===== EXPORT =====
comparison_path = os.path.join(RESULTS_DIR, "model_comparison_v1.csv")
preds_path      = os.path.join(RESULTS_DIR, "model_comparison_predictions.csv")
results_df.to_csv(comparison_path, index=False)
all_preds.to_csv(preds_path, index=False)

print(f"\n  ✅ {comparison_path}")
print(f"  ✅ {preds_path}")
