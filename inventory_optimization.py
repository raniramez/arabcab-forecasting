"""
ARABCAB Competition - Inventory Optimization

Calculate optimal inventory parameters for each material:
- Economic Order Quantity (EOQ)
- Safety Stock
- Reorder Point
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os

def calculate_eoq(annual_demand, ordering_cost, holding_cost_annual):
    """
    Calculate Economic Order Quantity
    
    EOQ = √(2 × D × S / H)
    
    Args:
        annual_demand: Annual demand in tons
        ordering_cost: Cost per order in USD
        holding_cost_annual: Holding cost in USD/ton/year
    
    Returns:
        EOQ in tons
    """
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_annual)
    return eoq

def calculate_safety_stock(demand_std_monthly, lead_time_months, service_level=0.95):
    """
    Calculate Safety Stock
    
    Safety Stock = Z × σ_demand × √L
    
    Args:
        demand_std_monthly: Standard deviation of monthly demand
        lead_time_months: Lead time in months
        service_level: Service level (e.g., 0.95 for 95%)
    
    Returns:
        Safety stock in tons
    """
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * demand_std_monthly * np.sqrt(lead_time_months)
    return safety_stock

def calculate_reorder_point(avg_monthly_demand, lead_time_months, safety_stock):
    """
    Calculate Reorder Point
    
    ROP = (Average monthly demand × Lead time) + Safety stock
    
    Args:
        avg_monthly_demand: Average monthly demand in tons
        lead_time_months: Lead time in months
        safety_stock: Safety stock in tons
    
    Returns:
        Reorder point in tons
    """
    rop = (avg_monthly_demand * lead_time_months) + safety_stock
    return rop

def calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost_annual):
    """
    Calculate total annual inventory cost
    
    Total Cost = (D/Q × S) + (Q/2 × H)
    where D = annual demand, Q = order quantity, S = ordering cost, H = holding cost
    
    Returns:
        Dictionary with cost breakdown
    """
    # Number of orders per year
    n_orders = annual_demand / eoq
    
    # Ordering cost
    annual_ordering_cost = n_orders * ordering_cost
    
    # Holding cost
    avg_inventory = eoq / 2
    annual_holding_cost = avg_inventory * holding_cost_annual
    
    total_cost = annual_ordering_cost + annual_holding_cost
    
    return {
        'total_cost': total_cost,
        'ordering_cost': annual_ordering_cost,
        'holding_cost': annual_holding_cost,
        'n_orders_per_year': n_orders,
        'avg_inventory_level': avg_inventory
    }

def optimize_inventory_for_material(material_name, forecast_df, 
                                    ordering_cost, holding_cost_monthly, 
                                    lead_time_months, service_level=0.95):
    """
    Calculate complete inventory optimization for one material
    
    Args:
        material_name: 'xlpe', 'pvc', or 'lsf'
        forecast_df: DataFrame with forecasts
        ordering_cost: Fixed cost per order (USD)
        holding_cost_monthly: Cost to hold 1 ton for 1 month (USD/ton/month)
        lead_time_months: Supplier lead time in months
        service_level: Target service level (default 95%)
    """
    print(f"\n{'='*60}")
    print(f"Inventory Optimization: {material_name.upper()}")
    print(f"{'='*60}")
    
    # Get forecasted demand
    forecast_col = f'{material_name}_forecast'
    forecasts = forecast_df[forecast_col].values
    
    # Calculate demand statistics
    avg_monthly_demand = np.mean(forecasts)
    demand_std = np.std(forecasts)
    annual_demand = avg_monthly_demand * 12
    
    # Convert monthly holding cost to annual
    holding_cost_annual = holding_cost_monthly * 12
    
    print(f"\nDemand Statistics:")
    print(f"  Average monthly demand: {avg_monthly_demand:.1f} tons")
    print(f"  Std deviation: {demand_std:.1f} tons")
    print(f"  Annual demand: {annual_demand:.1f} tons")
    
    print(f"\nParameters:")
    print(f"  Ordering cost: ${ordering_cost:.2f}/order")
    print(f"  Holding cost: ${holding_cost_monthly:.2f}/ton/month (${holding_cost_annual:.2f}/ton/year)")
    print(f"  Lead time: {lead_time_months} months")
    print(f"  Service level: {service_level*100:.0f}%")
    
    # Calculate EOQ
    eoq = calculate_eoq(annual_demand, ordering_cost, holding_cost_annual)
    
    # Calculate Safety Stock
    safety_stock = calculate_safety_stock(demand_std, lead_time_months, service_level)
    
    # Calculate Reorder Point
    reorder_point = calculate_reorder_point(avg_monthly_demand, lead_time_months, safety_stock)
    
    # Calculate costs
    cost_breakdown = calculate_total_cost(annual_demand, eoq, ordering_cost, holding_cost_annual)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"  Economic Order Quantity (EOQ): {eoq:.1f} tons")
    print(f"  Safety Stock: {safety_stock:.1f} tons")
    print(f"  Reorder Point: {reorder_point:.1f} tons")
    print(f"\nCost Analysis:")
    print(f"  Total annual cost: ${cost_breakdown['total_cost']:,.2f}")
    print(f"  - Ordering cost: ${cost_breakdown['ordering_cost']:,.2f}")
    print(f"  - Holding cost: ${cost_breakdown['holding_cost']:,.2f}")
    print(f"  Orders per year: {cost_breakdown['n_orders_per_year']:.1f}")
    print(f"  Average inventory: {cost_breakdown['avg_inventory_level']:.1f} tons")
    
    return {
        'material': material_name,
        'avg_monthly_demand': float(avg_monthly_demand),
        'demand_std': float(demand_std),
        'annual_demand': float(annual_demand),
        'eoq': float(eoq),
        'safety_stock': float(safety_stock),
        'reorder_point': float(reorder_point),
        'total_annual_cost': float(cost_breakdown['total_cost']),
        'ordering_cost_annual': float(cost_breakdown['ordering_cost']),
        'holding_cost_annual': float(cost_breakdown['holding_cost']),
        'orders_per_year': float(cost_breakdown['n_orders_per_year']),
        'avg_inventory_level': float(cost_breakdown['avg_inventory_level']),
        'parameters': {
            'ordering_cost': float(ordering_cost),
            'holding_cost_monthly': float(holding_cost_monthly),
            'lead_time_months': float(lead_time_months),
            'service_level': float(service_level)
        }
    }

def run_inventory_optimization():
    """Main inventory optimization pipeline"""
    print("="*60)
    print("ARABCAB Inventory Optimization")
    print("="*60)
    
    # Load forecasts
    forecast_df = pd.read_csv('results/forecasts.csv')
    print(f"\nLoaded forecasts: {len(forecast_df)} months")
    
    # Material-specific parameters (realistic industry values)
    materials_params = {
        'xlpe': {
            'ordering_cost': 500,  # USD per order
            'holding_cost_monthly': 15,  # USD/ton/month
            'lead_time_months': 2,  # 2 months lead time
        },
        'pvc': {
            'ordering_cost': 400,
            'holding_cost_monthly': 12,
            'lead_time_months': 1.5,
        },
        'lsf': {
            'ordering_cost': 600,
            'holding_cost_monthly': 20,
            'lead_time_months': 3,
        }
    }
    
    service_level = 0.95  # 95% service level
    
    all_results = {}
    
    for material, params in materials_params.items():
        results = optimize_inventory_for_material(
            material,
            forecast_df,
            params['ordering_cost'],
            params['holding_cost_monthly'],
            params['lead_time_months'],
            service_level
        )
        all_results[material] = results
    
    # Save results
    output_path = 'results/inventory_params.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Inventory optimization complete!")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("MATERIAL COMPARISON")
    print("="*60)
    print(f"{'Material':<10} {'EOQ (tons)':<15} {'Safety Stock':<15} {'ROP (tons)':<15} {'Annual Cost':<15}")
    print("-"*70)
    for material in ['xlpe', 'pvc', 'lsf']:
        r = all_results[material]
        print(f"{material.upper():<10} {r['eoq']:<15.1f} {r['safety_stock']:<15.1f} "
              f"{r['reorder_point']:<15.1f} ${r['total_annual_cost']:<14,.0f}")
    
    return all_results

if __name__ == "__main__":
    results = run_inventory_optimization()
    print("\n✓ Inventory optimization pipeline complete!")
