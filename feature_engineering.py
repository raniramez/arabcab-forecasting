"""
ARABCAB Competition - Feature Engineering

This script transforms raw demand data into ML-ready features including:
- Lag features
- Rolling statistics
- Time-based features
- Interaction features
"""

import pandas as pd
import numpy as np
import os

def create_lag_features(df, columns, lags=[1, 3, 6, 12]):
    """Create lag features for specified columns"""
    print(f"Creating lag features for: {columns}")
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df

def create_rolling_features(df, columns, windows=[3, 6, 12]):
    """Create rolling statistics features"""
    print(f"Creating rolling features for: {columns}")
    
    for col in columns:
        for window in windows:
            # Rolling mean
            df[f'{col}_rolling_mean{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling std (volatility measure)
            df[f'{col}_rolling_std{window}'] = df[col].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max for 12-month window only
            if window == 12:
                df[f'{col}_rolling_min{window}'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_max{window}'] = df[col].rolling(window=window, min_periods=1).max()
    
    return df

def create_time_features(df):
    """Create time-based features"""
    print("Creating time features...")
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Cyclical encoding of month (preserves cyclical nature)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Quarter
    df['quarter'] = df['date'].dt.quarter
    
    # Year trend (normalized)
    df['year_trend'] = (df['year'] - df['year'].min())
    
    # Month index from start
    df['month_index'] = np.arange(len(df))
    
    return df

def create_interaction_features(df):
    """Create interaction features between economic indicators"""
    print("Creating interaction features...")
    
    # Oil price × feedstock price
    df['oil_feedstock_interaction'] = df['crude_oil_price'] * df['polymer_feedstock_index'] / 100
    
    # Industrial production × construction index
    df['production_construction_interaction'] = (
        df['industrial_production_index'] * df['construction_index'] / 10000
    )
    
    # Demand/price ratio for each material (price sensitivity indicator)
    # Using oil price as proxy for input cost
    for material in ['xlpe', 'pvc', 'lsf']:
        demand_col = f'{material}_demand_tons'
        if demand_col in df.columns:
            df[f'{material}_demand_price_ratio'] = df[demand_col] / df['crude_oil_price']
    
    # Price volatility index (rolling std of oil price)
    df['price_volatility_index'] = df['crude_oil_price'].rolling(window=6, min_periods=1).std()
    
    # Supply chain stress indicator
    # (higher lead time + lower reliability = higher stress)
    df['supply_chain_stress'] = (
        (df['supplier_lead_time'] / df['supplier_lead_time'].max()) * 
        (1 - df['supplier_reliability'])
    )
    
    return df

def create_momentum_features(df, columns):
    """Create momentum/trend features"""
    print(f"Creating momentum features for: {columns}")
    
    for col in columns:
        # 3-month momentum (percent change over 3 months)
        df[f'{col}_momentum3'] = df[col].pct_change(periods=3) * 100
        
        # Rate of change (first difference)
        df[f'{col}_diff1'] = df[col].diff(1)
    
    return df

def engineer_features():
    """Main feature engineering pipeline"""
    print("="*60)
    print("ARABCAB Feature Engineering")
    print("="*60)
    
    # Load raw dataset
    input_path = 'data/raw_materials_demand_dataset.csv'
    print(f"\nLoading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    # Materials to create features for
    demand_columns = ['xlpe_demand_tons', 'pvc_demand_tons', 'lsf_demand_tons']
    economic_columns = ['crude_oil_price', 'polymer_feedstock_index', 
                       'industrial_production_index', 'construction_index']
    
    # 1. Lag features
    df = create_lag_features(df, demand_columns, lags=[1, 3, 6, 12])
    
    # 2. Rolling statistics
    df = create_rolling_features(df, demand_columns, windows=[3, 6, 12])
    df = create_rolling_features(df, economic_columns, windows=[3, 6])
    
    # 3. Time features
    df = create_time_features(df)
    
    # 4. Interaction features
    df = create_interaction_features(df)
    
    # 5. Momentum features
    df = create_momentum_features(df, demand_columns)
    df = create_momentum_features(df, economic_columns)
    
    # Fill NaN values created by lag/rolling operations
    # For lag features, we can't backfill, so drop initial rows or forward fill carefully
    print(f"\nNaN values before handling: {df.isnull().sum().sum()}")
    
    # Forward fill for initial NaN values (conservative approach)
    df = df.fillna(method='ffill')
    
    # For any remaining NaNs (e.g., first row), use backward fill
    df = df.fillna(method='bfill')
    
    # If still any NaNs, fill with 0 (shouldn't happen but just in case)
    df = df.fillna(0)
    
    print(f"NaN values after handling: {df.isnull().sum().sum()}")
    
    # Save engineered features
    output_path = 'data/features_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Feature engineering complete!")
    print(f"  Output saved to: {output_path}")
    print(f"  Final shape: {df.shape}")
    print(f"  Total features: {len(df.columns)}")
    
    print("\n" + "="*60)
    print("Feature Categories:")
    print("="*60)
    
    lag_features = [col for col in df.columns if 'lag' in col]
    rolling_features = [col for col in df.columns if 'rolling' in col]
    time_features = [col for col in df.columns if any(x in col for x in ['month_', 'quarter', 'year_trend', 'month_index'])]
    interaction_features = [col for col in df.columns if 'interaction' in col or 'ratio' in col or 'stress' in col]
    momentum_features = [col for col in df.columns if 'momentum' in col or 'diff' in col or 'volatility' in col]
    
    print(f"Lag features: {len(lag_features)}")
    print(f"Rolling features: {len(rolling_features)}")
    print(f"Time features: {len(time_features)}")
    print(f"Interaction features: {len(interaction_features)}")
    print(f"Momentum features: {len(momentum_features)}")
    
    return df

if __name__ == "__main__":
    df = engineer_features()
    print("\n✓ Feature engineering pipeline complete!")
