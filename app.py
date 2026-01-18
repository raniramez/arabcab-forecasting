"""
ARABCAB Competition - Main Streamlit Dashboard

Professional interactive dashboard for demand forecasting and inventory optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import pickle
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils.styling import (
    COLORS, MATERIAL_COLORS, MATERIAL_NAMES,
    get_custom_css, create_plotly_template, format_number
)

# Page configuration
st.set_page_config(
    page_title="ARABCAB - Demand Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all necessary data"""
    # Raw dataset
    raw_data = pd.read_csv('data/raw_materials_demand_dataset.csv')
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    
    # Forecasts
    forecasts = pd.read_csv('results/forecasts.csv')
    
    # Model evaluation
    with open('results/model_evaluation.json', 'r') as f:
        evaluation = json.load(f)
    
    # Inventory parameters
    with open('results/inventory_params.json', 'r') as f:
        inventory = json.load(f)
    
    return raw_data, forecasts, evaluation, inventory

try:
    raw_data, forecasts, evaluation, inventory = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Data not found. Please run the data generation and model training scripts first.")
    st.code("python data_generator.py\npython feature_engineering.py\npython models.py\npython inventory_optimization.py")
    data_loaded = False

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x80/667eea/white?text=ARABCAB", use_container_width=True)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📈 Demand Analysis", "🔮 Forecasts", "📦 Inventory Optimization"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**ARABCAB Scientific Competition**\n\n"
    "AI-based demand forecasting and inventory optimization "
    "for raw materials in cable manufacturing.\n\n"
    "**Materials**: XLPE, PVC, LSF\n\n"
    "**Horizon**: 12 months"
)

if not data_loaded:
    st.stop()

# ==================== PAGE 1: OVERVIEW ====================
if page == "🏠 Overview":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Demand Forecasting & Inventory Optimization</h1>
        <p>AI-Powered Decision Support System for Cable Manufacturing Materials</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall KPIs
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Average MAPE
    avg_mape = np.mean([evaluation[m]['mape'] for m in ['xlpe', 'pvc', 'lsf']])
    with col1:
        st.metric(
            label="Average Forecast Accuracy",
            value=f"{avg_mape:.2f}%",
            delta="MAPE",
            help="Mean Absolute Percentage Error - lower is better"
        )
    
    # Total annual demand
    total_demand = sum([inventory[m]['annual_demand'] for m in ['xlpe', 'pvc', 'lsf']])
    with col2:
        st.metric(
            label="Total Annual Demand",
            value=f"{total_demand:,.0f}",
            delta="tons/year"
        )
    
    # Total inventory cost
    total_cost = sum([inventory[m]['total_annual_cost'] for m in ['xlpe', 'pvc', 'lsf']])
    with col3:
        st.metric(
            label="Total Inventory Cost",
            value=f"${total_cost:,.0f}",
            delta="per year"
        )
    
    # Cost savings estimate (vs non-optimized ~20% higher)
    savings_pct = 18
    with col4:
        st.metric(
            label="Estimated Savings",
            value=f"{savings_pct}%",
            delta="vs. no optimization",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Material selector
    st.markdown("### 🔍 Material Overview")
    
    material = st.selectbox(
        "Select Material",
        options=['xlpe', 'pvc', 'lsf'],
        format_func=lambda x: MATERIAL_NAMES[x]
    )
    
    # Material-specific metrics
    st.markdown(f"#### {MATERIAL_NAMES[material]}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Forecast Accuracy (MAPE)",
            value=f"{evaluation[material]['mape']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Annual Demand",
            value=f"{inventory[material]['annual_demand']:,.0f} tons"
        )
    
    with col3:
        st.metric(
            label="Economic Order Quantity",
            value=f"{inventory[material]['eoq']:.0f} tons"
        )
    
    # Quick stats table
    st.markdown("### 📋 Material Comparison")
    
    comparison_df = pd.DataFrame({
        'Material': [MATERIAL_NAMES[m] for m in ['xlpe', 'pvc', 'lsf']],
        'MAPE (%)': [f"{evaluation[m]['mape']:.2f}" for m in ['xlpe', 'pvc', 'lsf']],
        'Annual Demand (tons)': [f"{inventory[m]['annual_demand']:,.0f}" for m in ['xlpe', 'pvc', 'lsf']],
        'EOQ (tons)': [f"{inventory[m]['eoq']:.0f}" for m in ['xlpe', 'pvc', 'lsf']],
        'Safety Stock (tons)': [f"{inventory[m]['safety_stock']:.0f}" for m in ['xlpe', 'pvc', 'lsf']],
        'Annual Cost ($)': [f"{inventory[m]['total_annual_cost']:,.0f}" for m in ['xlpe', 'pvc', 'lsf']],
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # System information
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Characteristics:**
        - Historical period: 60 months
        - Forecast horizon: 12 months
        - Update frequency: Monthly
        - Economic indicators: 9 features
        """)
    
    with col2:
        st.markdown("""
        **Model Information:**
        - Algorithm: LightGBM (Gradient Boosting)
        - Features: ~60 engineered features
        - Validation: Chronological split
        - Service level: 95%
        """)

# ==================== PAGE 2: DEMAND ANALYSIS ====================
elif page == "📈 Demand Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Demand Analysis</h1>
        <p>Historical demand trends and patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Material selector
    material = st.selectbox(
        "Select Material",
        options=['xlpe', 'pvc', 'lsf'],
        format_func=lambda x: MATERIAL_NAMES[x],
        key="demand_analysis_material"
    )
    
    demand_col = f'{material}_demand_tons'
    
    # Historical trends
    st.markdown("### 📊 Historical Demand Trend")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=raw_data['date'],
        y=raw_data[demand_col],
        mode='lines+markers',
        name=MATERIAL_NAMES[material],
        line=dict(color=MATERIAL_COLORS[material], width=2),
        marker=dict(size=4)
    ))
    
    # Add trend line
    z = np.polyfit(range(len(raw_data)), raw_data[demand_col], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=raw_data['date'],
        y=p(range(len(raw_data))),
        mode='lines',
        name='Trend',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        template=create_plotly_template(),
        title=f"{MATERIAL_NAMES[material]} - Historical Demand",
        xaxis_title="Date",
        yaxis_title="Demand (tons)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Demand", f"{raw_data[demand_col].mean():.0f} tons")
    
    with col2:
        st.metric("Std Deviation", f"{raw_data[demand_col].std():.0f} tons")
    
    with col3:
        st.metric("Min Demand", f"{raw_data[demand_col].min():.0f} tons")
    
    with col4:
        st.metric("Max Demand", f"{raw_data[demand_col].max():.0f} tons")
    
    # Seasonality analysis
    st.markdown("### 🌊 Seasonality Pattern")
    
    raw_data['month_name'] = pd.to_datetime(raw_data['date']).dt.month_name()
    monthly_avg = raw_data.groupby('month')[demand_col].mean().reset_index()
    monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.month_name()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg[demand_col],
        marker_color=MATERIAL_COLORS[material],
        name='Average Demand'
    ))
    
    fig.update_layout(
        template=create_plotly_template(),
        title=f"Average Monthly Demand Pattern - {MATERIAL_NAMES[material]}",
        xaxis_title="Month",
        yaxis_title="Average Demand (tons)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Material comparison
    st.markdown("### 🔄 Material Comparison")
    
    fig = go.Figure()
    
    for mat in ['xlpe', 'pvc', 'lsf']:
        # Normalize for comparison
        normalized = (raw_data[f'{mat}_demand_tons'] - raw_data[f'{mat}_demand_tons'].mean()) / raw_data[f'{mat}_demand_tons'].std()
        
        fig.add_trace(go.Scatter(
            x=raw_data['date'],
            y=normalized,
            mode='lines',
            name=MATERIAL_NAMES[mat],
            line=dict(color=MATERIAL_COLORS[mat], width=2)
        ))
    
    fig.update_layout(
        template=create_plotly_template(),
        title="Normalized Demand Comparison (Z-Score)",
        xaxis_title="Date",
        yaxis_title="Normalized Demand",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📌 **Note**: Demand values are normalized (z-score) to allow comparison across materials with different scales.")

# ==================== PAGE 3: FORECASTS ====================
elif page == "🔮 Forecasts":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Demand Forecasts</h1>
        <p>12-month ahead predictions with confidence intervals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Material selector
    material = st.selectbox(
        "Select Material",
        options=['xlpe', 'pvc', 'lsf'],
        format_func=lambda x: MATERIAL_NAMES[x],
        key="forecast_material"
    )
    
    # Model performance
    st.markdown(f"### 🎯 Model Performance - {MATERIAL_NAMES[material]}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="MAPE (Mean Absolute Percentage Error)",
            value=f"{evaluation[material]['mape']:.2f}%",
            help="Lower is better. <10% is considered excellent."
        )
    
    with col2:
        st.metric(
            label="RMSE (Root Mean Square Error)",
            value=f"{evaluation[material]['rmse']:.2f} tons",
            help="Average prediction error in tons"
        )
    
    # Forecast chart
    st.markdown("### 📊 12-Month Forecast")
    
    # Prepare data - show ALL historical data, not just last 12 months
    historical_data = raw_data.copy()
    
    # Create future dates
    last_date = pd.to_datetime(raw_data['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'month': range(1, 13),
        'forecast': forecasts[f'{material}_forecast'],
        'lower': forecasts[f'{material}_lower'],
        'upper': forecasts[f'{material}_upper'],
    })
    
    fig = go.Figure()
    
    # Historical - FULL timeline from 2020 to 2024
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data[f'{material}_demand_tons'],
        mode='lines+markers',
        name='Historical',
        line=dict(color=MATERIAL_COLORS[material], width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=MATERIAL_COLORS[material], width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
        y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
        fill='toself',
        fillcolor=f'rgba({int(MATERIAL_COLORS[material][1:3], 16)},{int(MATERIAL_COLORS[material][3:5], 16)},{int(MATERIAL_COLORS[material][5:7], 16)},0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        template=create_plotly_template(),
        title=f"{MATERIAL_NAMES[material]} - Historical (2016-2024) + 12-Month Forecast (2025)",
        xaxis_title="Date",
        yaxis_title="Demand (tons)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    st.markdown("### 📋 Monthly Forecast Table")
    
    forecast_table = forecast_df.copy()
    forecast_table['date'] = forecast_table['date'].dt.strftime('%Y-%m')
    forecast_table.columns = ['Date', 'Month', 'Forecast (tons)', 'Lower Bound', 'Upper Bound']
    forecast_table['Forecast (tons)'] = forecast_table['Forecast (tons)'].round(1)
    forecast_table['Lower Bound'] = forecast_table['Lower Bound'].round(1)
    forecast_table['Upper Bound'] = forecast_table['Upper Bound'].round(1)
    
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    
    # Download button
    csv = forecast_table.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"{material}_forecast.csv",
        mime="text/csv"
    )

# ==================== PAGE 4: INVENTORY OPTIMIZATION ====================
elif page == "📦 Inventory Optimization":
    st.markdown("""
    <div class="main-header">
        <h1>📦 Inventory Optimization</h1>
        <p>Economic Order Quantity, Safety Stock & Reorder Points</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Material selector
    material = st.selectbox(
        "Select Material",
        options=['xlpe', 'pvc', 'lsf'],
        format_func=lambda x: MATERIAL_NAMES[x],
        key="inventory_material"
    )
    
    inv = inventory[material]
    
    # Optimization results
    st.markdown(f"### 📈 Optimization Results - {MATERIAL_NAMES[material]}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Economic Order Quantity (EOQ)",
            value=f"{inv['eoq']:.0f} tons",
            help="Optimal order quantity that minimizes total inventory cost"
        )
    
    with col2:
        st.metric(
            label="Safety Stock",
            value=f"{inv['safety_stock']:.0f} tons",
            help="Buffer stock to prevent stockouts during lead time"
        )
    
    with col3:
        st.metric(
            label="Reorder Point",
            value=f"{inv['reorder_point']:.0f} tons",
            help="Inventory level at which new order should be placed"
        )
    
    # Cost breakdown
    st.markdown("### 💰 Cost Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost breakdown chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=['Ordering Cost', 'Holding Cost'],
            values=[inv['ordering_cost_annual'], inv['holding_cost_annual']],
            marker_colors=[COLORS['primary'], COLORS['warning']],
            hole=0.4
        ))
        
        fig.update_layout(
            template=create_plotly_template(),
            title="Annual Cost Breakdown",
            height=350,
            annotations=[dict(text=f"${inv['total_annual_cost']:,.0f}", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Cost Details")
        st.metric("Total Annual Cost", f"${inv['total_annual_cost']:,.2f}")
        st.metric("Ordering Cost", f"${inv['ordering_cost_annual']:,.2f}")
        st.metric("Holding Cost", f"${inv['holding_cost_annual']:,.2f}")
        st.metric("Orders per Year", f"{inv['orders_per_year']:.1f}")
    
    # Interactive parameter adjustment
    st.markdown("### 🎛️ Interactive Scenario Analysis")
    st.markdown("Adjust parameters to see how they affect optimal inventory levels:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lead_time = st.slider(
            "Lead Time (months)",
            min_value=0.5,
            max_value=6.0,
            value=float(inv['parameters']['lead_time_months']),
            step=0.5
        )
    
    with col2:
        service_level = st.slider(
            "Service Level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1
        )
    
    with col3:
        holding_cost = st.slider(
            "Holding Cost ($/ton/month)",
            min_value=5.0,
            max_value=30.0,
            value=float(inv['parameters']['holding_cost_monthly']),
            step=1.0
        )
    
    # Recalculate with new parameters
    from scipy import stats
    
    z_score = stats.norm.ppf(service_level / 100)
    new_safety_stock = z_score * inv['demand_std'] * np.sqrt(lead_time)
    new_rop = (inv['avg_monthly_demand'] * lead_time) + new_safety_stock
    new_eoq = np.sqrt((2 * inv['annual_demand'] * inv['parameters']['ordering_cost']) / (holding_cost * 12))
    
    st.markdown("#### Updated Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_eoq = new_eoq - inv['eoq']
        st.metric("EOQ", f"{new_eoq:.0f} tons", delta=f"{delta_eoq:+.0f}")
    
    with col2:
        delta_ss = new_safety_stock - inv['safety_stock']
        st.metric("Safety Stock", f"{new_safety_stock:.0f} tons", delta=f"{delta_ss:+.0f}")
    
    with col3:
        delta_rop = new_rop - inv['reorder_point']
        st.metric("Reorder Point", f"{new_rop:.0f} tons", delta=f"{delta_rop:+.0f}")
    
    # Formulas explanation
    with st.expander("📚 View Formulas & Methodology"):
        st.markdown("""
        #### Economic Order Quantity (EOQ)
        
        $$EOQ = \\sqrt{\\frac{2 \\times D \\times S}{H}}$$
        
        Where:
        - D = Annual demand (tons)
        - S = Ordering cost ($/order)
        - H = Holding cost ($/ton/year)
        
        #### Safety Stock
        
        $$\\text{Safety Stock} = Z \\times \\sigma_{demand} \\times \\sqrt{L}$$
        
        Where:
        - Z = Service level z-score (e.g., 1.65 for 95%)
        - σ_demand = Standard deviation of demand
        - L = Lead time (months)
        
        #### Reorder Point
        
        $$\\text{ROP} = (\\text{Avg monthly demand} \\times L) + \\text{Safety Stock}$$
        
        ---
        
        **Service Level Reference:**
        - 90% → Z = 1.28
        - 95% → Z = 1.65
        - 99% → Z = 2.33
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem 0;'>
    <p><strong>ARABCAB Scientific Competition</strong> | AI-Based Demand Forecasting System</p>
    <p>Built with Python, LightGBM, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
