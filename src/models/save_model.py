import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "V2", "hv_france_weekly_features_v2.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "src", "api", "saved_model")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    'year', 'week_of_year', 'month', 'quarter',
    'is_summer', 'is_winter', 'is_soldes', 'is_high_season',
    'Famille_encoded',
    'lag_1', 'lag_2', 'lag_4', 'lag_52',
    'rolling_mean_4', 'rolling_mean_12'
]
TARGET = 'sum_Quantite'

print("Loading data...")
df = pd.read_csv(INPUT_PATH, parse_dates=['week_start'])

# Filter same as training
min_avg_sales = 500
high_volume   = df.groupby('Famille')['sum_Quantite'].mean()
high_volume   = high_volume[high_volume >= min_avg_sales].index.tolist()
df = df[df['Famille'].isin(high_volume)]

exclude = ['BIJOUX', 'CHAUSSURE', 'DIVERS', 'CUIR PAM',
           'MAROQUINERIE', 'NAF NAF', 'GROSSES PIÈCES']
df = df[~df['Famille'].isin(exclude)]

# Save Famille → encoded number mapping
famille_cat     = df['Famille'].astype('category')
famille_map_inv = {v: k for k, v in dict(enumerate(famille_cat.cat.categories)).items()}

# Train on all data before 2024
train   = df[df['week_start'] < '2024-01-01']
X_train = train[FEATURES].values
y_train = train[TARGET].values

# Scale + train
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model          = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Save all artifacts
joblib.dump(model,           os.path.join(MODELS_DIR, "ridge_model.pkl"))
joblib.dump(scaler,          os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(famille_map_inv, os.path.join(MODELS_DIR, "famille_map.pkl"))
joblib.dump(FEATURES,        os.path.join(MODELS_DIR, "features.pkl"))

print("✅ Model saved!")
print(f"   Families supported : {sorted(famille_map_inv.keys())}")
print(f"   Training rows      : {len(train):,}")
print(f"   Saved to           : {MODELS_DIR}")
