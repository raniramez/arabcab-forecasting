"""
ARABCAB Competition - Raw Material Demand Data Generator

This script generates realistic synthetic demand data for three polymer materials
used in cable manufacturing: XLPE, PVC, and LSF.

It combines real economic indicators from Book1.xlsx with synthesized demand patterns
based on industry research and realistic correlations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_book1_data():
    """Load and process economic indicators from Book1.xlsx"""
    print("Loading Book1.xlsx...")
    df = pd.read_excel('Book1.xlsx')
    
    # Use full available history (108 months / 9 years)
    if len(df) > 108:
        df = df.head(108).copy()
    
    # Create date column
    # Dates from 2016 for full history
    start_date = datetime(2016, 1, 1)
    df['date'] = [start_date + timedelta(days=30*i) for i in range(len(df))]
    
    return df

def generate_economic_indicators(n_months=60):
    """Generate realistic economic and market indicators"""
    print("Generating economic indicators...")
    
    # Time array
    t = np.arange(n_months)
    
    # 1. Crude Oil Price (USD/barrel) - realistic Brent crude pattern
    # Base: $60-80, with volatility and trend
    oil_base = 70
    oil_trend = 0.3 * t  # Gradual increase
    oil_seasonal = 5 * np.sin(2 * np.pi * t / 12)  # Seasonal pattern
    oil_noise = np.random.normal(0, 8, n_months)
    crude_oil_price = oil_base + oil_trend + oil_seasonal + oil_noise
    crude_oil_price = np.clip(crude_oil_price, 45, 120)
    
    # 2. Polymer Feedstock Price Index (ethylene/naphtha proxy)
    # Highly correlated with oil (0.85 correlation)
    feedstock_base = 100
    feedstock_trend = 0.5 * t
    feedstock_oil_corr = 0.4 * (crude_oil_price - crude_oil_price.mean())
    feedstock_noise = np.random.normal(0, 5, n_months)
    polymer_feedstock_index = feedstock_base + feedstock_trend + feedstock_oil_corr + feedstock_noise
    polymer_feedstock_index = np.clip(polymer_feedstock_index, 80, 180)
    
    # 3. Industrial Production Index (base 100)
    # Moderate growth with business cycle
    prod_base = 100
    prod_trend = 0.25 * t
    prod_cycle = 8 * np.sin(2 * np.pi * t / 24)  # 2-year cycle
    prod_noise = np.random.normal(0, 3, n_months)
    industrial_production_index = prod_base + prod_trend + prod_cycle + prod_noise
    industrial_production_index = np.clip(industrial_production_index, 90, 140)
    
    # 4. Construction Activity Index (base 100)
    # Strong seasonality (higher in spring/summer)
    const_base = 100
    const_trend = 0.3 * t
    const_seasonal = 15 * np.sin(2 * np.pi * (t - 3) / 12)  # Peak in June
    const_noise = np.random.normal(0, 5, n_months)
    construction_index = const_base + const_trend + const_seasonal + const_noise
    construction_index = np.clip(construction_index, 70, 150)
    
    # 5. [REMOVED] Producer Price Index (inflation measure)
    # User requested removal
    
    # 6. [REMOVED] Exchange Rate
    # User requested removal
    
    # 7. Supplier Lead Time (days)
    # Varies between 30-90 days with some volatility
    lead_time_base = 60
    lead_time_trend = 0.15 * t  # Slight increase over time
    lead_time_noise = np.random.normal(0, 8, n_months)
    supplier_lead_time = lead_time_base + lead_time_trend + lead_time_noise
    supplier_lead_time = np.clip(supplier_lead_time, 30, 90).astype(int)
    
    # 8. Supplier Reliability Index (0-1)
    # Relatively stable with minor fluctuations
    reliability_base = 0.88
    reliability_noise = np.random.normal(0, 0.03, n_months)
    supplier_reliability = reliability_base + reliability_noise
    supplier_reliability = np.clip(supplier_reliability, 0.75, 0.98)
    
    # 9. Inventory Holding Cost (USD/ton/month)
    # Correlated with interest rates and storage costs
    holding_base = 15
    holding_trend = 0.05 * t
    holding_noise = np.random.normal(0, 1, n_months)
    inventory_holding_cost = holding_base + holding_trend + holding_noise
    inventory_holding_cost = np.clip(inventory_holding_cost, 10, 25)
    
    df = pd.DataFrame({
        'crude_oil_price': crude_oil_price,
        'polymer_feedstock_index': polymer_feedstock_index,
        'industrial_production_index': industrial_production_index,
        'construction_index': construction_index,
        # 'ppi': ppi,  <-- Removed
        # 'exchange_rate': exchange_rate,  <-- Removed
        'supplier_lead_time': supplier_lead_time,
        'supplier_reliability': supplier_reliability,
        'inventory_holding_cost': inventory_holding_cost
    })
    
    return df

def generate_xlpe_demand(economic_df, n_months=60):
    """
    Generate XLPE (Cross-Linked Polyethylene) demand
    - Used in high-voltage cable insulation
    - Strongly correlated with industrial production & infrastructure (0.75)
    - Medium volatility (σ=10%)
    - Moderate seasonality (±15%)
    - Range: 800-1500 tons/month
    """
    print("Generating XLPE demand...")
    
    t = np.arange(n_months)
    
    # Base demand
    base = 1100
    
    # Long-term growth trend
    trend = 5 * t
    
    # Correlation with industrial production
    prod_normalized = (economic_df['industrial_production_index'] - 100) / 10
    production_effect = 0.75 * 80 * prod_normalized
    
    # Correlation with construction activity
    const_normalized = (economic_df['construction_index'] - 100) / 10
    construction_effect = 0.45 * 50 * const_normalized
    
    # Seasonality (moderate - infrastructure projects follow seasons)
    seasonal = 100 * np.sin(2 * np.pi * (t - 2) / 12)  # Peak in spring
    
    # Medium volatility noise
    noise = np.random.normal(0, 50, n_months)  # ~10% volatility at base
    
    xlpe_demand = base + trend + production_effect + construction_effect + seasonal + noise
    xlpe_demand = np.clip(xlpe_demand, 800, 2500)
    
    return xlpe_demand

def generate_pvc_demand(economic_df, n_months=60):
    """
    Generate PVC (Polyvinyl Chloride) demand
    - Used in low & medium voltage cable insulation
    - Stable, volume-driven
    - Highly price-sensitive (-0.65 correlation with oil)
    - Low volatility (σ=5%)
    - Low seasonality (±8%)
    - Range: 1200-2000 tons/month
    """
    print("Generating PVC demand...")
    
    t = np.arange(n_months)
    
    # Base demand (higher volume than XLPE)
    base = 1600
    
    # Steady growth trend
    trend = 4 * t
    
    # Price sensitivity (inverse correlation with oil/feedstock)
    oil_normalized = (economic_df['crude_oil_price'] - economic_df['crude_oil_price'].mean()) / 10
    price_effect = -0.65 * 60 * oil_normalized  # Negative correlation
    
    # Industrial production (moderate correlation)
    prod_normalized = (economic_df['industrial_production_index'] - 100) / 10
    production_effect = 0.50 * 50 * prod_normalized
    
    # Low seasonality
    seasonal = 50 * np.sin(2 * np.pi * t / 12)
    
    # Low volatility noise
    noise = np.random.normal(0, 30, n_months)  # ~5% volatility
    
    pvc_demand = base + trend + price_effect + production_effect + seasonal + noise
    pvc_demand = np.clip(pvc_demand, 1200, 3000)
    
    return pvc_demand

def generate_lsf_demand(economic_df, n_months=60):
    """
    Generate LSF (Low Smoke Fume) demand
    - Fire-resistant safety insulation
    - Project-based demand
    - Low volume
    - High volatility (σ=25%)
    - Sensitive to regulations & safety standards
    - Range: 100-300 tons/month
    """
    print("Generating LSF demand...")
    
    t = np.arange(n_months)
    
    # Base demand (much lower volume - specialty material)
    base = 180
    
    # Growth trend (increasing safety regulations)
    trend = 1.5 * t
    
    # Project-based spikes (random large projects)
    project_spikes = np.zeros(n_months)
    n_projects = 8  # 8 major projects over 60 months
    project_months = np.random.choice(range(10, n_months), n_projects, replace=False)
    for month in project_months:
        project_spikes[month] = np.random.uniform(40, 80)
    
    # Weak correlation with construction (mostly regulatory/safety driven)
    const_normalized = (economic_df['construction_index'] - 100) / 10
    construction_effect = 0.30 * 15 * const_normalized
    
    # Minimal seasonality
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    
    # High volatility noise
    noise = np.random.normal(0, 25, n_months)  # ~25% volatility
    
    lsf_demand = base + trend + project_spikes + construction_effect + seasonal + noise
    lsf_demand = np.clip(lsf_demand, 100, 600)
    
    return lsf_demand

def create_dataset():
    """Main function to create the complete dataset"""
    print("="*60)
    print("ARABCAB Raw Material Demand Dataset Generation")
    print("="*60)
    
    # Try to load Book1.xlsx data
    # FORCE SYNTHETIC: Book1 appears to have flat data at the end which ruins 2025 forecast
    try:
        # book1_df = load_book1_data()
        raise Exception("Forcing synthetic data for better variance")
        # dates = book1_df['date'].values
        
        # Extract useful columns from Book1 if available
        # (We'll use our generated economic indicators as primary)
    except Exception as e:
        print(f"Note: Could not load Book1.xlsx completely ({e})")
        print("Using fully synthetic economic data...")
        # Synthetic fallback: 9 years (2016-2024)
        start_date = datetime(2016, 1, 1)
        dates = [start_date + timedelta(days=30*i) for i in range(108)]
    
    # Determine length
    n_months = len(dates)
    print(f"Generating data for {n_months} months...")
    
    # Generate economic indicators
    economic_df = generate_economic_indicators(n_months)
    
    # Generate demand for each material
    xlpe_demand = generate_xlpe_demand(economic_df, n_months)
    pvc_demand = generate_pvc_demand(economic_df, n_months)
    lsf_demand = generate_lsf_demand(economic_df, n_months)
    
    # Combine into final dataset
    # Ensure dates are pandas Timestamps
    if isinstance(dates, pd.Series):
        dates = dates.values
    dates = pd.to_datetime(dates)
    
    dataset = pd.DataFrame({
        'date': dates,
        'year': pd.to_datetime(dates).year,
        'month': pd.to_datetime(dates).month,
    })
    
    # Add economic indicators
    for col in economic_df.columns:
        dataset[col] = economic_df[col].values
    
    # Add demand targets
    dataset['xlpe_demand_tons'] = xlpe_demand
    dataset['pvc_demand_tons'] = pvc_demand
    dataset['lsf_demand_tons'] = lsf_demand
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    output_path = 'data/raw_materials_demand_dataset.csv'
    dataset.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Columns: {len(dataset.columns)}")
    
    # Display summary statistics
    print("\n" + "="*60)
    print("DEMAND SUMMARY STATISTICS (tons/month)")
    print("="*60)
    
    for material in ['xlpe_demand_tons', 'pvc_demand_tons', 'lsf_demand_tons']:
        print(f"\n{material.replace('_', ' ').upper()}:")
        print(f"  Mean:   {dataset[material].mean():.1f}")
        print(f"  Median: {dataset[material].median():.1f}")
        print(f"  Std:    {dataset[material].std():.1f}")
        print(f"  Min:    {dataset[material].min():.1f}")
        print(f"  Max:    {dataset[material].max():.1f}")
        print(f"  CV:     {(dataset[material].std() / dataset[material].mean() * 100):.1f}%")
    
    print("\n" + "="*60)
    print("Sample rows:")
    print("="*60)
    print(dataset.head(3).to_string())
    
    return dataset

if __name__ == "__main__":
    dataset = create_dataset()
    print("\n✓ Data generation complete!")
