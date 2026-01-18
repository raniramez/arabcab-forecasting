"""
ARABCAB Competition - Dashboard Styling Utilities

Centralized styling and theme configuration for professional appearance
"""

import plotly.graph_objects as go
import plotly.express as px

# Color Palette
COLORS = {
    'primary': '#1f77b4',
    'xlpe': '#2ca02c',  # Green - infrastructure
    'pvc': '#ff7f0e',   # Orange - construction
    'lsf': '#d62728',   # Red - safety/fire
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'light_gray': '#e9ecef',
    'dark_gray': '#6c757d',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
}

MATERIAL_COLORS = {
    'xlpe': COLORS['xlpe'],
    'pvc': COLORS['pvc'],
    'lsf': COLORS['lsf'],
}

MATERIAL_NAMES = {
    'xlpe': 'XLPE (High-Voltage Insulation)',
    'pvc': 'PVC (Low/Medium Voltage)',
    'lsf': 'LSF (Fire-Resistant Safety)',
}

def get_custom_css():
    """Return custom CSS for Streamlit app"""
    return """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Material badges */
    .material-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    
    .badge-xlpe {
        background-color: #d4edda;
        color: #155724;
    }
    
    .badge-pvc {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .badge-lsf {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
    }
    
    /* Metric containers */
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe thead tr th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
    """

def create_plotly_template():
    """Create custom Plotly template for consistent chart styling"""
    template = go.layout.Template()
    
    template.layout = go.Layout(
        font=dict(family="Inter, sans-serif", size=12, color=COLORS['text']),
        title=dict(font=dict(size=18, color=COLORS['text'], family="Roboto, sans-serif")),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
        margin=dict(t=60, l=60, r=40, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e9ecef',
            showline=True,
            linecolor='#dee2e6',
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e9ecef',
            showline=True,
            linecolor='#dee2e6',
            zeroline=False,
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#dee2e6',
            borderwidth=1,
        )
    )
    
    return template

def format_number(value, prefix='', suffix='', decimals=0):
    """Format numbers for display"""
    if decimals == 0:
        return f"{prefix}{value:,.0f}{suffix}"
    else:
        return f"{prefix}{value:,.{decimals}f}{suffix}"

def create_metric_card(label, value, delta=None, delta_color="normal"):
    """Create HTML for a metric card"""
    delta_html = ""
    if delta is not None:
        color = COLORS['success'] if delta_color == "normal" else COLORS['danger']
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.25rem;">{delta}</div>'
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
