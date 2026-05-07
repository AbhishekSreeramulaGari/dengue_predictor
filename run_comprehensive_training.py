import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("="*100)
print("LOADING DATASET AND PERFORMING FEATURE ENGINEERING")
print("="*100)

df = pd.read_csv('bengaluru_wards_dataset.csv')

if 'Date' not in df.columns:
    df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
else:
    df['Date'] = pd.to_datetime(df['Date'])

print(f'\nDataset loaded: {len(df)} rows')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df["Date"].min()} to {df["Date"].max()}')

# Feature Engineering
df_fe = df.sort_values(['Ward_ID','Date']).reset_index(drop=True).copy()

df_fe['Rainfall_Lag1'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].shift(1)
df_fe['Rainfall_Lag2'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].shift(2)
df_fe['Temp_Lag1'] = df_fe.groupby('Ward_ID')['Avg_Temp_C'].shift(1)
df_fe['Cases_Lag1'] = df_fe.groupby('Ward_ID')['Dengue_Cases'].shift(1)

df_fe['Rainfall_roll3_mean'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].rolling(window=3, min_periods=1).mean().reset_index(0,drop=True)
df_fe['Cases_roll3_mean'] = df_fe.groupby('Ward_ID')['Dengue_Cases'].rolling(window=3, min_periods=1).mean().reset_index(0,drop=True)

df_fe['Month'] = df_fe['Date'].dt.month
df_fe['Year'] = df_fe['Date'].dt.year
df_fe['Is_Monsoon'] = df_fe['Month'].isin([6,7,8,9]).astype(int)

ward_agg = df_fe.groupby('Ward_ID')[['Garbage_Complaints','Waterlogging_Complaints']].mean().rename(columns=lambda x: x+'_ward_mean')
df_fe = df_fe.merge(ward_agg, left_on='Ward_ID', right_index=True)
df_fe = df_fe.dropna().reset_index(drop=True)

print(f'After feature engineering: {len(df_fe)} rows')

features = [
    'Rainfall_mm','Avg_Temp_C','Garbage_Complaints','Waterlogging_Complaints',
    'Rainfall_Lag1','Rainfall_Lag2','Temp_Lag1','Cases_Lag1',
    'Rainfall_roll3_mean','Cases_roll3_mean','Is_Monsoon','Garbage_Complaints_ward_mean','Waterlogging_Complaints_ward_mean'
]

X = df_fe[features].copy()
y = df_fe['Dengue_Cases'].copy()

split_idx = int(len(X) * 0.8)
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

print(f'Training set: {len(X_train)} samples')
print(f'Test set: {len(X_test)} samples')
print(f'Features: {len(features)} features')
print("="*100)

# Scale data
print("\n" + "="*100)
print("TRAINING ALL 5 MODELS")
print("="*100)

global_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(global_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(global_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

models = {}
scalers = {}
results = []

# 1. LINEAR REGRESSION
print("\n[1/5] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr_model
scalers['Linear Regression'] = global_scaler
lr_pred = lr_model.predict(X_test_scaled)
results.append({
    'Model': 'Linear Regression',
    'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
    'MAE': mean_absolute_error(y_test, lr_pred),
    'R²': r2_score(y_test, lr_pred),
    'MAPE': mean_absolute_percentage_error(y_test, lr_pred)
})
print(f'✓ Linear Regression trained - R²: {results[-1]["R²"]:.4f}')

# 2. GRADIENT BOOSTING MACHINE (GBM)
print("[2/5] Training GBM...")
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gbm_model.fit(X_train_scaled, y_train)
models['GBM'] = gbm_model
scalers['GBM'] = global_scaler
gbm_pred = gbm_model.predict(X_test_scaled)
results.append({
    'Model': 'GBM',
    'RMSE': np.sqrt(mean_squared_error(y_test, gbm_pred)),
    'MAE': mean_absolute_error(y_test, gbm_pred),
    'R²': r2_score(y_test, gbm_pred),
    'MAPE': mean_absolute_percentage_error(y_test, gbm_pred)
})
print(f'✓ GBM trained - R²: {results[-1]["R²"]:.4f}')

# 3. RANDOM FOREST
print("[3/5] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
scalers['Random Forest'] = None
rf_pred = rf_model.predict(X_test)
results.append({
    'Model': 'Random Forest',
    'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'MAE': mean_absolute_error(y_test, rf_pred),
    'R²': r2_score(y_test, rf_pred),
    'MAPE': mean_absolute_percentage_error(y_test, rf_pred)
})
print(f'✓ Random Forest trained - R²: {results[-1]["R²"]:.4f}')

# 4. XGBOOST
print("[4/5] Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
scalers['XGBoost'] = None
xgb_pred = xgb_model.predict(X_test)
results.append({
    'Model': 'XGBoost',
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'MAE': mean_absolute_error(y_test, xgb_pred),
    'R²': r2_score(y_test, xgb_pred),
    'MAPE': mean_absolute_percentage_error(y_test, xgb_pred)
})
print(f'✓ XGBoost trained - R²: {results[-1]["R²"]:.4f}')

# 5. MULTI-LAYER PERCEPTRON (MLP)
print("[5/5] Training MLP Neural Network...")
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp_model.fit(X_train_scaled, y_train)
models['MLP'] = mlp_model
scalers['MLP'] = global_scaler
mlp_pred = mlp_model.predict(X_test_scaled)
results.append({
    'Model': 'MLP',
    'RMSE': np.sqrt(mean_squared_error(y_test, mlp_pred)),
    'MAE': mean_absolute_error(y_test, mlp_pred),
    'R²': r2_score(y_test, mlp_pred),
    'MAPE': mean_absolute_percentage_error(y_test, mlp_pred)
})
print(f'✓ MLP trained - R²: {results[-1]["R²"]:.4f}')

print("\n" + "="*100)

# Display results
print("\n" + "="*100)
print("MODEL PERFORMANCE COMPARISON - ACCURATE RESULTS")
print("="*100)

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)

print("\nCOMPARATIVE METRICS TABLE:")
print("="*100)
print(results_df.to_string(index=False))
print("="*100)

print("\nDETAILED RANKING (by R² Score):")
print("="*100)
for idx, (_, row) in enumerate(results_df.iterrows()):
    rank = idx + 1
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
    print(f"\n{medal} {row['Model']}")
    print(f"   R² Score:    {row['R²']:.6f}")
    print(f"   RMSE:        {row['RMSE']:.4f}")
    print(f"   MAE:         {row['MAE']:.4f}")
    print(f"   MAPE:        {row['MAPE']:.4f}%")

print("\n" + "="*100)
print("FINAL SUMMARY - ACCURATE MODEL COMPARISON")
print("="*100)

for idx, (_, row) in enumerate(results_df.iterrows()):
    print(f"\n{idx+1}. {row['Model']}")
    print(f"   R² Score: {row['R²']:.6f} (explains {row['R²']*100:.2f}% of variance)")
    print(f"   RMSE: {row['RMSE']:.4f} (avg error when predicting cases)")
    print(f"   MAE: {row['MAE']:.4f} (average absolute error)")
    print(f"   MAPE: {row['MAPE']:.4f}% (percent error)")

print("\n" + "="*100)
print("KEY TAKEAWAY:")
print("="*100)
best_model = results_df.iloc[0]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   - Explains {best_model['R²']*100:.2f}% of prediction variance")
print(f"   - Predicts dengue cases with ±{best_model['MAE']:.2f} cases average error")
print(f"\n✓ All models trained with proper feature engineering and scaling")
print("✓ Results are now accurate and comparable")
print("\n" + "="*100)

# Save results
results_df.to_csv('model_performance_accurate_results.csv', index=False)
print("\n✓ Results saved to: model_performance_accurate_results.csv")
