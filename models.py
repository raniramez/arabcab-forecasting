"""
ARABCAB Competition - Forecasting Models

Train separate LightGBM models for each material (XLPE, PVC, LSF)
and generate 12-month forecasts with confidence intervals.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pickle
import json
import os

def prepare_data_for_material(df, material):
    """
    Prepare features and target for a specific material
    
    Args:
        df: DataFrame with all features
        material: 'xlpe', 'pvc', or 'lsf'
    
    Returns:
        X (features), y (target)
    """
    target_col = f'{material}_demand_tons'
    
    # Exclude columns that shouldn't be used as features
    exclude_cols = [
        'date', 'year', 'month',  # Time identifiers
        'xlpe_demand_tons', 'pvc_demand_tons', 'lsf_demand_tons',  # All targets
    ]
    
    # Remove demand/price ratios and momentum for OTHER materials
    # (keep only the current material's derived features)
    other_materials = [m for m in ['xlpe', 'pvc', 'lsf'] if m != material]
    for other_mat in other_materials:
        exclude_cols.extend([col for col in df.columns if col.startswith(f'{other_mat}_')])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y, feature_cols

def train_model(X_train, y_train, X_val, y_val, material_name):
    """Train LightGBM model with early stopping"""
    print(f"\n{'='*40}")
    print(f"TRAINING SEPARATE MODEL FOR: {material_name.upper()}")
    print(f"{'='*40}")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',  # Changed back to standard regression for variability
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    return model

def evaluate_model(model, X_test, y_test, material_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{material_name.upper()} Model Performance:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f} tons")
    
    return {'mape': mape, 'rmse': rmse, 'predictions': y_pred}

def generate_future_forecast(model, df, feature_cols, material, n_months=12):
    """
    Generate future forecasts with confidence intervals
    
    Projects economic indicators forward using trends and seasonality,
    then iteratively predicts demand while updating lag features.
    """
    print(f"\nGenerating 12-month forecast for {material.upper()}...")
    
    # Get recent historical data for trend analysis
    recent_data = df.tail(12).copy()
    
    # Project economic indicators forward (simplified trend + seasonality)
    def project_indicator(col_name, n_future):
        """Project an indicator forward using trend and seasonality"""
        if col_name not in df.columns:
            return None
        
        values = df[col_name].values[-12:]  # Last year
        
        # Simple trend (linear)
        trend = (values[-1] - values[0]) / 12
        
        # Simple seasonality (average pattern)
        seasonal_pattern = values - values.mean()
        
        projections = []
        for i in range(n_future):
            # Trend component
            trend_value = values[-1] + trend * (i + 1)
            
            # Seasonal component (repeat pattern)
            seasonal_idx = i % 12
            seasonal_value = seasonal_pattern[seasonal_idx] if seasonal_idx < len(seasonal_pattern) else 0
            
            # Small random noise for realism
            # Force minimum noise if std is too low
            std_val = max(values.std(), values.mean() * 0.02)
            noise = np.random.normal(0, std_val * 0.2)
            
            # Full seasonality + noise
            projected = trend_value + seasonal_value * 1.0 + noise
            projections.append(projected)
        
        return projections
    
    # Project key economic indicators
    economic_indicators = [
        'crude_oil_price', 'polymer_feedstock_index', 'industrial_production_index',
        'construction_index', 'supplier_lead_time',
        'supplier_reliability', 'inventory_holding_cost'
    ]
    
    projected_econ = {}
    for indicator in economic_indicators:
        if indicator in df.columns:
            projected_econ[indicator] = project_indicator(indicator, n_months)
    
    # Now generate forecasts iteratively
    forecasts = []
    lower_bounds = []
    upper_bounds = []
    
    # Start with last row's features
    current_features = df[feature_cols].iloc[-1].copy()
    last_demand = df[f'{material}_demand_tons'].iloc[-1]
    
    for i in range(n_months):
        # Update economic indicators with stronger variation
        for indicator in economic_indicators:
            if indicator in projected_econ and projected_econ[indicator] is not None:
                # Update base indicator
                if indicator in current_features.index:
                    current_features[indicator] = projected_econ[indicator][i]
                
                # Update rolling features for this indicator
                for window in [3, 6]:
                    rolling_mean_col = f'{indicator}_rolling_mean{window}'
                    if rolling_mean_col in current_features.index:
                        current_features[rolling_mean_col] = projected_econ[indicator][i]
        
        # Update time features
        if 'month_index' in current_features.index:
            current_features['month_index'] += 1
        
        # Update month cyclical encoding
        current_month = (int(df['month'].iloc[-1]) + i) % 12 + 1
        if 'month_sin' in current_features.index:
            current_features['month_sin'] = np.sin(2 * np.pi * current_month / 12)
        if 'month_cos' in current_features.index:
            current_features['month_cos'] = np.cos(2 * np.pi * current_month / 12)
        
        # Update lag features (use previous predictions)
        if i > 0:
            if f'{material}_demand_tons_lag1' in current_features.index:
                current_features[f'{material}_demand_tons_lag1'] = last_demand
        
        if i >= 3:
            if f'{material}_demand_tons_lag3' in current_features.index:
                current_features[f'{material}_demand_tons_lag3'] = forecasts[i-3]
        
        if i >= 6:
            if f'{material}_demand_tons_lag6' in current_features.index:
                current_features[f'{material}_demand_tons_lag6'] = forecasts[i-6]
        
        # Update rolling demand features
        if i >= 3:
            recent_forecasts = forecasts[max(0, i-3):i]
            if f'{material}_demand_tons_rolling_mean3' in current_features.index:
                current_features[f'{material}_demand_tons_rolling_mean3'] = np.mean(recent_forecasts) if recent_forecasts else last_demand
        
        # Update interaction features
        if 'oil_feedstock_interaction' in current_features.index:
            oil = current_features.get('crude_oil_price', 70)
            feedstock = current_features.get('polymer_feedstock_index', 100)
            current_features['oil_feedstock_interaction'] = oil * feedstock / 100
        
        if 'production_construction_interaction' in current_features.index:
            prod = current_features.get('industrial_production_index', 100)
            const = current_features.get('construction_index', 100)
            current_features['production_construction_interaction'] = prod * const / 10000
        
        # Make prediction
        pred = model.predict(current_features.values.reshape(1, -1))[0]
        
        forecasts.append(pred)
        
        # Confidence intervals based on material volatility
        historical_demand = df[f'{material}_demand_tons'].values
        residual_std = historical_demand.std() * 0.15
        lower_bounds.append(pred - 1.96 * residual_std)
        upper_bounds.append(pred + 1.96 * residual_std)
        
        # Update last_demand for next iteration
        last_demand = pred
    
    return forecasts, lower_bounds, upper_bounds

def train_all_models():
    """Main training pipeline"""
    print("="*60)
    print("ARABCAB Forecasting Models Training")
    print("="*60)
    
    # Load engineered features
    df = pd.read_csv('data/features_engineered.csv')
    print(f"\nLoaded data shape: {df.shape}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Train/test split (chronological)
    # First 48 months: train
    # Months 49-54: validation
    # Months 55-60: test
    # Dynamic split based on data length
    test_size = 6
    val_size = 6
    train_size = len(df) - val_size - test_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} months")
    print(f"  Val: {len(val_df)} months")
    print(f"  Test: {len(test_df)} months")
    
    materials = ['xlpe', 'pvc', 'lsf']
    models = {}
    evaluations = {}
    all_forecasts = {}
    
    for material in materials:
        print(f"\n{'='*60}")
        print(f"Processing {material.upper()}")
        print(f"{'='*60}")
        
        # Prepare data
        X_train, y_train, feature_cols = prepare_data_for_material(train_df, material)
        X_val, y_val, _ = prepare_data_for_material(val_df, material)
        X_test, y_test, _ = prepare_data_for_material(test_df, material)
        
        print(f"Features: {len(feature_cols)}")
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, material)
        
        # Evaluate
        eval_results = evaluate_model(model, X_test, y_test, material)
        
        # Generate future forecast
        forecasts, lower, upper = generate_future_forecast(
            model, df, feature_cols, material, n_months=12
        )
        
        # Store results
        models[material] = model
        evaluations[material] = {
            'mape': eval_results['mape'],
            'rmse': eval_results['rmse']
        }
        all_forecasts[material] = {
            'forecast': forecasts,
            'lower_bound': lower,
            'upper_bound': upper
        }
        
        # Save model
        model_path = f'models/{material}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved to: {model_path}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.1f}")
    
    # Save evaluation metrics
    eval_path = 'results/model_evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(evaluations, f, indent=2)
    print(f"\n✓ Evaluation metrics saved to: {eval_path}")
    
    # Save forecasts
    forecast_df = pd.DataFrame({
        'month': range(1, 13),
        'xlpe_forecast': all_forecasts['xlpe']['forecast'],
        'xlpe_lower': all_forecasts['xlpe']['lower_bound'],
        'xlpe_upper': all_forecasts['xlpe']['upper_bound'],
        'pvc_forecast': all_forecasts['pvc']['forecast'],
        'pvc_lower': all_forecasts['pvc']['lower_bound'],
        'pvc_upper': all_forecasts['pvc']['upper_bound'],
        'lsf_forecast': all_forecasts['lsf']['forecast'],
        'lsf_lower': all_forecasts['lsf']['lower_bound'],
        'lsf_upper': all_forecasts['lsf']['upper_bound'],
    })
    
    forecast_path = 'results/forecasts.csv'
    forecast_df.to_csv(forecast_path, index=False)
    print(f"✓ Forecasts saved to: {forecast_path}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("\nModel Performance:")
    for material in materials:
        print(f"  {material.upper()}: MAPE={evaluations[material]['mape']:.2f}%, "
              f"RMSE={evaluations[material]['rmse']:.2f} tons")
    
    avg_mape = np.mean([evaluations[m]['mape'] for m in materials])
    print(f"\nAverage MAPE: {avg_mape:.2f}%")
    
    return models, evaluations, all_forecasts

if __name__ == "__main__":
    models, evaluations, forecasts = train_all_models()
    print("\n✓ Model training complete!")
