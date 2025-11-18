"""
Trading IA Dashboard
Dashboard interactivo para monitoreo y análisis de estrategias de trading
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="Trading IA Dashboard",
    page_icon="📈",
    layout="wide"
)

# Título
st.title("📈 Trading IA Dashboard")
st.markdown("---")

# Sidebar con navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Seleccionar página:", [
    "Resumen Ejecutivo",
    "Análisis de Estrategias",
    "Backtests Recientes",
    "Monitoreo en Vivo",
    "Configuración"
])

# Función para cargar datos
@st.cache_data
def load_strategy_results():
    """Cargar resultados de estrategias"""
    try:
        with open('results/strategy_rankings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_backtest_results():
    """Cargar resultados de backtests"""
    try:
        with open('results/backtest_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Página principal - Resumen Ejecutivo
if page == "Resumen Ejecutivo":
    st.header("📊 Resumen Ejecutivo")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    strategy_data = load_strategy_results()
    backtest_data = load_backtest_results()

    if strategy_data:
        # Mejor estrategia por Sharpe
        best_sharpe = strategy_data['sharpe'].split('\n')[1].split('|')[1].strip()
        with col1:
            st.metric("Mejor Estrategia (Sharpe)", best_sharpe)

    if backtest_data:
        # Último backtest
        latest_backtest = backtest_data[0]
        with col2:
            st.metric("Último Retorno", ".1f")
        with col3:
            st.metric("Último Sharpe", ".2f")
        with col4:
            st.metric("Total Trades", latest_backtest['metrics']['total_trades'])

    # Gráfico de rendimiento
    st.subheader("Rendimiento por Estrategia")
    if strategy_data:
        # Parsear datos de Sharpe
        lines = strategy_data['sharpe'].split('\n')[1:-1]  # Skip header and empty line
        strategies = []
        sharpes = []

        for line in lines[:10]:  # Top 10
            parts = line.split('|')
            if len(parts) >= 3:
                strategy = f"{parts[1].strip()} {parts[2].strip()}"
                sharpe = float(parts[3].strip())
                strategies.append(strategy)
                sharpes.append(sharpe)

        fig = go.Figure(data=[
            go.Bar(x=strategies, y=sharpes, marker_color='lightblue')
        ])
        fig.update_layout(
            title="Top 10 Estrategias por Ratio Sharpe",
            xaxis_title="Estrategia",
            yaxis_title="Ratio Sharpe"
        )
        st.plotly_chart(fig, use_container_width=True)

# Página de análisis de estrategias
elif page == "Análisis de Estrategias":
    st.header("🔍 Análisis de Estrategias")

    if strategy_data := load_strategy_results():
        # Tabs para diferentes vistas
        tab1, tab2, tab3 = st.tabs(["Por Sharpe", "Por Win Rate", "Overall Score"])

        with tab1:
            st.subheader("Ranking por Ratio Sharpe")
            st.code(strategy_data['sharpe'], language='text')

        with tab2:
            st.subheader("Ranking por Tasa de Ganancia")
            st.code(strategy_data['win_rate'], language='text')

        with tab3:
            st.subheader("Ranking General")
            st.code(strategy_data['overall'], language='text')
    else:
        st.warning("No se encontraron datos de estrategias")

# Página de backtests recientes
elif page == "Backtests Recientes":
    st.header("📈 Backtests Recientes")

    if backtest_data := load_backtest_results():
        for i, backtest in enumerate(backtest_data[:5]):  # Mostrar últimos 5
            with st.expander(f"Backtest {i+1}: {backtest['config']['strategy_name']} - {backtest['timestamp'][:10]}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Configuración:**")
                    st.json(backtest['config'])

                with col2:
                    st.write("**Métricas:**")
                    metrics = backtest['metrics']
                    st.write(".2f")
                    st.write(".2f")
                    st.write(".2f")
                    st.write(".2f")
                    st.write(f"Trades: {metrics['total_trades']}")
                    st.write(".1f")
    else:
        st.warning("No se encontraron resultados de backtests")

# Página de monitoreo en vivo
elif page == "Monitoreo en Vivo":
    st.header("📡 Monitoreo en Vivo")

    st.info("Funcionalidad de monitoreo en vivo próximamente disponible")

    # Placeholder para métricas en vivo
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PnL Actual", "$0.00", "0.00%")

    with col2:
        st.metric("Trades Activos", "0")

    with col3:
        st.metric("Estado", "Detenido")

# Página de configuración
elif page == "Configuración":
    st.header("⚙️ Configuración")

    st.subheader("Parámetros de Trading")
    risk_per_trade = st.slider("Riesgo por Trade (%)", 0.1, 5.0, 2.0)
    max_trades = st.slider("Máximo Trades Abiertos", 1, 10, 3)

    if st.button("Guardar Configuración"):
        config = {
            "risk_per_trade": risk_per_trade / 100,
            "max_open_trades": max_trades,
            "updated_at": datetime.now().isoformat()
        }

        with open('config/dashboard_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        st.success("Configuración guardada exitosamente!")

# Footer
st.markdown("---")
st.markdown("*Dashboard generado automáticamente - Trading IA System*")