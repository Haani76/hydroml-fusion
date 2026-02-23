"""
HydroML-Fusion Interactive Dashboard - Complete Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Page config
st.set_page_config(
    page_title="HydroML-Fusion Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Header
st.markdown("# 🌊 HydroML-Fusion Dashboard")
st.markdown("**Multi-Model Ensemble Hydrological Forecasting System**")
st.markdown("*Leaf River Basin, Mississippi | 2010-2020*")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View:",
    ["🏠 Overview", "📊 Model Performance", "🌍 Geospatial Impact", 
     "🌡️ Climate Scenarios", "📉 Uncertainty", "🗺️ Maps", "📈 Results Summary"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Basin:** Leaf River, MS  
**Area:** 1,950 km²  
**Period:** 2010-2020
""")

# Load data
@st.cache_data
def load_data():
    base_path = Path(__file__).parent.parent / "data"
    
    data = {}
    data['gr4j'] = pd.read_csv(base_path / "processed/gr4j_validation_results.csv", parse_dates=['date'])
    data['xgb'] = pd.read_csv(base_path / "processed/xgboost_validation_results.csv", parse_dates=['date'])
    data['lstm'] = pd.read_csv(base_path / "processed/lstm_validation_results.csv", parse_dates=['date'])
    data['lstm_geo'] = pd.read_csv(base_path / "processed/lstm_geospatial_results.csv", parse_dates=['date'])
    data['ensemble'] = pd.read_csv(base_path / "processed/ensemble_results.csv", parse_dates=['date'])
    data['uncertainty'] = pd.read_csv(base_path / "forecasts/probabilistic_forecasts.csv", parse_dates=['date'])
    data['climate_summary'] = pd.read_csv(base_path / "forecasts/climate_impact_summary.csv")
    data['ssp245'] = pd.read_csv(base_path / "forecasts/scenario_ssp2_4.5_projections.csv", parse_dates=['date'])
    data['ssp585'] = pd.read_csv(base_path / "forecasts/scenario_ssp5_8.5_projections.csv", parse_dates=['date'])
    
    return data

try:
    data = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error: {e}")
    data_loaded = False

# PAGE 1: OVERVIEW
if page == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", "LSTM+Geo", "NSE: 0.928")
    with col2:
        st.metric("Ensemble NSE", "0.915", "")
    with col3:
        st.metric("Geospatial Gain", "+0.023", "")
    with col4:
        st.metric("Climate Impact", "-1.8%", "SSP2-4.5")
    
    if data_loaded:
        st.markdown("---")
        st.subheader("📊 Model Performance")
        
        models = ['GR4J', 'XGBoost', 'LSTM', 'LSTM+Geo', 'Ensemble']
        nse_values = [0.729, 0.737, 0.905, 0.928, 0.915]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=models, y=nse_values,
            text=[f"{v:.3f}" for v in nse_values],
            textposition='outside',
            marker_color=['#ff7f0e', '#2ca02c', '#d62728', '#1f77b4', '#9467bd']
        ))
        
        fig.update_layout(
            title="Nash-Sutcliffe Efficiency (NSE)",
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PAGE 2: MODEL PERFORMANCE
elif page == "📊 Model Performance":
    if data_loaded:
        model = st.selectbox("Select Model:", ["Ensemble", "LSTM+Geospatial", "LSTM", "XGBoost", "GR4J"])
        
        fig = go.Figure()
        
        if model == "Ensemble":
            df = data['ensemble']
            fig.add_trace(go.Scatter(x=df['date'], y=df['observed'], name='Observed', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['ensemble'], name='Ensemble', line=dict(color='red', width=2, dash='dash')))
        
        elif model == "LSTM+Geospatial":
            df = data['lstm_geo']
            fig.add_trace(go.Scatter(x=df['date'], y=df['observed'], name='Observed', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['predicted'], name='LSTM+Geo', line=dict(color='green', width=2, dash='dash')))
        
        elif model == "LSTM":
            df = data['lstm']
            fig.add_trace(go.Scatter(x=df['date'], y=df['observed'], name='Observed', line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['predicted'], name='LSTM', line=dict(color='blue', width=2, dash='dash')))
        
        fig.update_layout(
            title=f"Streamflow - {model}",
            xaxis_title="Date",
            yaxis_title="Streamflow (mm/day)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PAGE 3: GEOSPATIAL IMPACT
elif page == "🌍 Geospatial Impact":
    st.subheader("Impact of Adding Geospatial Features")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LSTM Baseline", "NSE: 0.905", "6 features")
    with col2:
        st.metric("LSTM + Geospatial", "NSE: 0.928", "+0.023 improvement")
    
    if data_loaded:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("LSTM Baseline", "LSTM + Geospatial")
        )
        
        df_base = data['lstm']
        fig.add_trace(go.Scatter(x=df_base['date'], y=df_base['observed'], name='Observed', line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_base['date'], y=df_base['predicted'], name='Baseline', line=dict(color='blue', dash='dash')), row=1, col=1)
        
        df_geo = data['lstm_geo']
        fig.add_trace(go.Scatter(x=df_geo['date'], y=df_geo['observed'], name='Observed', line=dict(color='black'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_geo['date'], y=df_geo['predicted'], name='+Geospatial', line=dict(color='green', dash='dash')), row=2, col=1)
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Added Features:**
    - Elevation (91m)
    - Slope (1.28°)
    - TWI (8.41)
    - Forest % (65.3%)
    - Agriculture % (14.7%)
    - Wetland % (1.6%)
    - Developed % (3.2%)
    """)

# PAGE 4: CLIMATE SCENARIOS
elif page == "🌡️ Climate Scenarios":
    st.subheader("Future Climate Projections (2041-2060)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### SSP2-4.5")
        st.metric("Temperature", "+1.8°C")
        st.metric("Streamflow", "-1.8%", "↓")
    with col2:
        st.markdown("### SSP5-8.5")
        st.metric("Temperature", "+2.7°C")
        st.metric("Streamflow", "-1.3%", "↓")
    
    if data_loaded:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['ssp245']['date'], 
            y=data['ssp245']['predicted_streamflow_mm'],
            name='SSP2-4.5',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=data['ssp585']['date'], 
            y=data['ssp585']['predicted_streamflow_mm'],
            name='SSP5-8.5',
            line=dict(color='red')
        ))
        
        fig.add_hline(y=2.92, line_dash="dash", annotation_text="Historical (2.92 mm/day)")
        
        fig.update_layout(
            title="Future Streamflow Under Climate Scenarios",
            xaxis_title="Date",
            yaxis_title="Streamflow (mm/day)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PAGE 5: UNCERTAINTY
elif page == "📉 Uncertainty":
    st.subheader("Probabilistic Forecasts (Monte Carlo Dropout)")
    
    if data_loaded:
        df = data['uncertainty']
        
        fig = go.Figure()
        
        # Observed
        fig.add_trace(go.Scatter(x=df['date'], y=df['observed'], name='Observed', line=dict(color='black', width=2)))
        
        # Mean
        fig.add_trace(go.Scatter(x=df['date'], y=df['mean_prediction'], name='Mean', line=dict(color='blue', width=2)))
        
        # 90% CI
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['p95'],
            fill=None, mode='lines',
            line_color='lightblue',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['p05'],
            fill='tonexty', mode='lines',
            line_color='lightblue',
            name='90% CI'
        ))
        
        fig.update_layout(
            title="Uncertainty Bands",
            xaxis_title="Date",
            yaxis_title="Streamflow (mm/day)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Std Dev", f"{df['std_prediction'].mean():.2f} mm/day")
        with col2:
            st.metric("90% CI Width", f"{(df['p95'] - df['p05']).mean():.2f} mm/day")
        with col3:
            coverage = ((df['observed'] >= df['p05']) & (df['observed'] <= df['p95'])).mean() * 100
            st.metric("Coverage", f"{coverage:.1f}%")

# PAGE 6: MAPS
elif page == "🗺️ Maps":
    st.subheader("Geospatial Data")
    
    st.markdown("### Digital Elevation Model (DEM)")
    st.markdown("- **Source:** SRTM 90m (OpenTopography)")
    st.markdown("- **Range:** 12m - 213m")
    st.markdown("- **Mean:** 103m")
    
    st.markdown("### Terrain Features")
    st.markdown("- **Slope:** 0° - 17.5° (mean: 1.94°)")
    st.markdown("- **TWI:** Topographic Wetness Index calculated")
    
    st.markdown("### Land Use")
    st.markdown("""
    - Forest: 65.3%
    - Agriculture: 14.7%
    - Wetland: 1.6%
    - Developed: 3.2%
    """)

# PAGE 7: SUMMARY
elif page == "📈 Results Summary":
    st.subheader("Project Achievements")
    
    st.markdown("""
    ### 🏆 Key Results
    - **Best Model:** LSTM + Geospatial (NSE = 0.928)
    - **Geospatial Impact:** +0.023 NSE improvement
    - **Ensemble:** NSE = 0.915
    - **Climate Scenarios:** -1.8% streamflow (SSP2-4.5)
    - **Uncertainty:** Monte Carlo Dropout (100 samples)
    """)
    
    st.markdown("### 📊 All Models")
    
    results = pd.DataFrame({
        'Model': ['GR4J', 'XGBoost', 'LSTM', 'LSTM+Geo', 'Ensemble'],
        'NSE': [0.729, 0.737, 0.905, 0.928, 0.915],
        'RMSE': [1.185, 0.974, 0.592, 0.514, 0.558]
    })
    
    st.dataframe(results, hide_index=True)
    
    st.markdown("""
    ### 🛠️ Technologies
    - PyTorch (LSTM)
    - XGBoost
    - FastAPI
    - Docker
    - NetCDF processing
    - Real SRTM DEM
    """)

st.markdown("---")
st.markdown("**HydroML-Fusion** | GitHub: @Haani76")