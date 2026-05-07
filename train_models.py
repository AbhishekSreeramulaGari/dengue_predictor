import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import joblib


def main():
    df = pd.read_csv('bengaluru_wards_synthetic.csv')
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    else:
        df['Date'] = pd.to_datetime(df['Date'])

    df_fe = df.sort_values(['Ward_ID', 'Date']).reset_index(drop=True).copy()
    df_fe['Rainfall_Lag1'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].shift(1)
    df_fe['Rainfall_Lag2'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].shift(2)
    df_fe['Temp_Lag1'] = df_fe.groupby('Ward_ID')['Avg_Temp_C'].shift(1)
    df_fe['Cases_Lag1'] = df_fe.groupby('Ward_ID')['Dengue_Cases'].shift(1)
    df_fe['Rainfall_roll3_mean'] = df_fe.groupby('Ward_ID')['Rainfall_mm'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df_fe['Cases_roll3_mean'] = df_fe.groupby('Ward_ID')['Dengue_Cases'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df_fe['Month'] = df_fe['Date'].dt.month
    df_fe['Year'] = df_fe['Date'].dt.year
    df_fe['Is_Monsoon'] = df_fe['Month'].isin([6,7,8,9]).astype(int)
    ward_agg = df_fe.groupby('Ward_ID')[['Garbage_Complaints','Waterlogging_Complaints']].mean().rename(columns=lambda x: x+'_ward_mean')
    df_fe = df_fe.merge(ward_agg, left_on='Ward_ID', right_index=True)
    df_fe = df_fe.dropna().reset_index(drop=True)

    features = [
        'Rainfall_mm','Avg_Temp_C','Garbage_Complaints','Waterlogging_Complaints',
        'Rainfall_Lag1','Rainfall_Lag2','Temp_Lag1','Cases_Lag1',
        'Rainfall_roll3_mean','Cases_roll3_mean','Is_Monsoon','Garbage_Complaints_ward_mean','Waterlogging_Complaints_ward_mean'
    ]
    target = 'Dengue_Cases'

    X = df_fe[features].copy()
    y = df_fe[target].copy()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    joblib.dump(scaler, 'scaler.joblib')

    X_tree_train, X_tree_test = X_train, X_test
    X_lin_train, X_lin_test = X_train_scaled, X_test_scaled

    models = {}
    try:
        import xgboost as xgb
        models['xgboost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    except Exception:
        print('XGBoost not installed — skipping XGBoost model')

    models['random_forest'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    models['gbm'] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    models['linear'] = LinearRegression()
    models['mlp'] = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)

    results = []
    fitted_models = {}
    for name, model in models.items():
        print(f'Training {name}')
        if name in ['linear','mlp']:
            Xtr, Xte = X_lin_train, X_lin_test
        else:
            Xtr, Xte = X_tree_train, X_tree_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f'{name} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}')
        results.append({'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2})
        fitted_models[name] = model
        joblib.dump(model, f'model_{name}.joblib')

    res_df = pd.DataFrame(results).sort_values('rmse')
    print('\nSummary:')
    print(res_df)

    # Print saved artifacts
    import os
    artifacts = [f for f in os.listdir('.') if f.startswith('model_') and f.endswith('.joblib')]
    if os.path.exists('scaler.joblib'):
        artifacts.append('scaler.joblib')
    print('Saved artifacts:', artifacts)


if __name__ == '__main__':
    main()
