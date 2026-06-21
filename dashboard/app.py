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
import yaml
from pathlib import Path

# Import authentication
try:
    from dashboard.auth import check_password, show_logout_button
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
RULES_DIR = PROJECT_ROOT / "core" / "rules"

# Configuración de la página
st.set_page_config(
    page_title="Trading IA Dashboard",
    page_icon="📈",
    layout="wide"
)

# Authentication check
if AUTH_AVAILABLE:
    if not check_password():
        st.stop()
    show_logout_button()

# Título
st.title("📈 Trading IA Dashboard")
st.markdown("---")

# Sidebar con navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Seleccionar página:", [
    "Resumen Ejecutivo",
    "Análisis de Estrategias",
    "Optimización (Council)",
    "Backtests Recientes",
    "Monitoreo en Vivo",
    "Configuración"
])

# Función para cargar datos
@st.cache_data
def load_strategy_results():
    """Cargar resultados de estrategias"""
    try:
        with open(RESULTS_DIR / 'strategy_rankings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_backtest_results():
    """Cargar resultados de backtests"""
    try:
        with open(RESULTS_DIR / 'backtest_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

@st.cache_data
def load_optimization_results():
    """Cargar resultados de optimización con Council"""
    try:
        with open(RESULTS_DIR / 'optimization_council_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_risk_rules():
    """Cargar reglas de riesgo desde YAML"""
    try:
        with open(RULES_DIR / 'risk_limits.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

# Página principal - Resumen Ejecutivo
if page == "Resumen Ejecutivo":
    st.header("📊 Resumen Ejecutivo")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    strategy_data = load_strategy_results()
    backtest_data = load_backtest_results()
    opt_data = load_optimization_results()

    if strategy_data:
        # Mejor estrategia por Sharpe
        best_sharpe = strategy_data['sharpe'].split('\n')[1].split('|')[1].strip()
        with col1:
            st.metric("Mejor Estrategia (Sharpe)", best_sharpe)

    if backtest_data:
        # Último backtest
        latest_backtest = backtest_data[0]
        with col2:
            st.metric("Último Retorno", f"{latest_backtest['metrics']['total_return']:.1%}")
        with col3:
            st.metric("Último Sharpe", f"{latest_backtest['metrics']['sharpe_ratio']:.2f}")
        with col4:
            st.metric("Total Trades", latest_backtest['metrics']['total_trades'])
            
    if opt_data:
        st.markdown("### Estado de Optimización (Council)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Robustness Score", f"{opt_data['robustness_score']:.3f}")
        with c2:
            st.metric("Avg Test Calmar", f"{opt_data['avg_test_calmar']:.3f}")

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

elif page == "Optimización (Council)":
    st.header("🧬 Optimización Walk-Forward con Council")
    
    data = load_optimization_results()
    
    if not data:
        st.warning("No se encontraron resultados de optimización. Ejecute 'scripts/optimize_strategy_with_council.py' primero.")
    else:
        # 1. Mejores Parámetros Globales
        st.subheader("🏆 Mejores Parámetros Globales")
        params = data['best_params_overall']
        
        # Display params in columns
        cols = st.columns(len(params))
        for i, (k, v) in enumerate(params.items()):
            with cols[i]:
                st.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
                
        st.markdown("---")
        
        # 2. Análisis por Periodo
        st.subheader("📅 Análisis Walk-Forward por Periodo")
        
        periods = data['periods']
        
        # Prepare DataFrame for display
        period_data = []
        for p in periods:
            metrics = p['test_metrics']
            row = {
                'Periodo': p['period'],
                'Train Start': p['train_start'],
                'Test Start': p['test_start'],
                'Test End': p['test_end'],
                'Calmar': metrics['calmar_ratio'],
                'Sharpe': metrics['sharpe_ratio'],
                'Return': metrics['total_return'],
                'Drawdown': metrics['max_drawdown'],
                'Trades': metrics['total_trades'],
                'Win Rate': metrics['win_rate']
            }
            period_data.append(row)
            
        df_periods = pd.DataFrame(period_data)
        
        # Format columns
        st.dataframe(df_periods.style.format({
            'Calmar': '{:.2f}',
            'Sharpe': '{:.2f}',
            'Return': '{:.2%}',
            'Drawdown': '{:.2%}',
            'Win Rate': '{:.2%}'
        }))
        
        # 3. Gráficos de Estabilidad
        st.subheader("📈 Estabilidad de Métricas")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_periods['Periodo'], y=df_periods['Calmar'], name='Calmar Ratio', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df_periods['Periodo'], y=df_periods['Sharpe'], name='Sharpe Ratio', mode='lines+markers'))
        
        fig.update_layout(title="Evolución de Métricas OOS (Out-of-Sample)", xaxis_title="Periodo", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Evolución de Parámetros
        st.subheader("🔄 Evolución de Parámetros Óptimos")
        
        param_keys = periods[0]['best_params'].keys()
        selected_param = st.selectbox("Seleccionar Parámetro", list(param_keys))
        
        param_values = [p['best_params'][selected_param] for p in periods]
        
        fig_param = go.Figure()
        fig_param.add_trace(go.Scatter(x=df_periods['Periodo'], y=param_values, name=selected_param, mode='lines+markers', line=dict(color='orange')))
        fig_param.update_layout(title=f"Estabilidad de {selected_param}", xaxis_title="Periodo", template="plotly_dark")
        st.plotly_chart(fig_param, use_container_width=True)

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

    # Auto-refresh configuration
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Intervalo (segundos)", 5, 60, 10)
    
    if auto_refresh:
        st.sidebar.info(f"Actualizando cada {refresh_interval}s")
        # Use st.empty() for dynamic content
        import time
        placeholder = st.empty()
        
    # Risk Manager Alerts Section
    st.subheader("🚨 Alertas del Risk Manager")
    
    # Load active alerts
    alerts_file = RESULTS_DIR / 'risk_alerts.json'
    try:
        if alerts_file.exists():
            with open(alerts_file, 'r') as f:
                alerts = json.load(f)
            
            if alerts:
                for alert in alerts[-10:]:  # Last 10 alerts
                    severity = alert.get('severity', 'INFO')
                    if severity == 'CRITICAL':
                        st.error(f"🔴 {alert['timestamp']}: {alert['message']}")
                    elif severity == 'WARNING':
                        st.warning(f"🟡 {alert['timestamp']}: {alert['message']}")
                    else:
                        st.info(f"🔵 {alert['timestamp']}: {alert['message']}")
            else:
                st.success("✅ No hay alertas activas")
        else:
            st.info("No hay historial de alertas")
    except Exception as e:
        st.warning(f"Error cargando alertas: {e}")
    
    st.markdown("---")
    
    # Live metrics
    st.subheader("📊 Métricas en Tiempo Real")
    col1, col2, col3, col4 = st.columns(4)

    # Try to load live status
    live_status_file = RESULTS_DIR / 'live_status.json'
    try:
        if live_status_file.exists():
            with open(live_status_file, 'r') as f:
                live_status = json.load(f)
            
            with col1:
                pnl = live_status.get('pnl', 0)
                pnl_pct = live_status.get('pnl_pct', 0)
                st.metric("PnL Actual", f"${pnl:.2f}", f"{pnl_pct:.2%}")
            
            with col2:
                st.metric("Trades Activos", live_status.get('active_trades', 0))
            
            with col3:
                drawdown = live_status.get('current_drawdown', 0)
                st.metric("Drawdown", f"{drawdown:.2%}")
            
            with col4:
                status = live_status.get('status', 'Detenido')
                st.metric("Estado", status)
        else:
            with col1:
                st.metric("PnL Actual", "$0.00", "0.00%")
            with col2:
                st.metric("Trades Activos", "0")
            with col3:
                st.metric("Drawdown", "0.00%")
            with col4:
                st.metric("Estado", "Detenido")
    except Exception as e:
        st.warning(f"Error cargando estado: {e}")

    # Kill Switch Status
    st.markdown("---")
    st.subheader("🛡️ Estado del Kill Switch")
    
    kill_switch_file = RESULTS_DIR / 'kill_switch_status.json'
    try:
        if kill_switch_file.exists():
            with open(kill_switch_file, 'r') as f:
                ks_status = json.load(f)
            
            if ks_status.get('active', False):
                st.error(f"🔴 KILL SWITCH ACTIVO - Razón: {ks_status.get('reason', 'N/A')}")
                st.write(f"Activado: {ks_status.get('activated_at', 'N/A')}")
            else:
                st.success("🟢 Kill Switch Inactivo - Trading habilitado")
        else:
            st.info("Estado del Kill Switch no disponible")
    except Exception as e:
        st.warning(f"Error: {e}")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Página de configuración
elif page == "Configuración":
    st.header("⚙️ Configuración")

    st.subheader("Reglas de Riesgo (Council)")
    rules_data = load_risk_rules()
    
    if rules_data and 'rules' in rules_data:
        for rule in rules_data['rules']:
            with st.expander(f"Regla: {rule['id']} ({rule['severity']})"):
                st.markdown(f"**Descripción:** {rule['description']}")
                st.markdown(f"**Condición:** `{rule['metric']} {rule['operator']} {rule['value']}`")
                st.markdown(f"**Acción:** {rule['action']}")
    else:
        st.warning("No se pudieron cargar las reglas de riesgo.")

    st.markdown("---")
    st.subheader("Parámetros de Dashboard")
    risk_per_trade = st.slider("Riesgo por Trade (%)", 0.1, 5.0, 2.0)
    max_trades = st.slider("Máximo Trades Abiertos", 1, 10, 3)

    if st.button("Guardar Configuración"):
        config = {
            "risk_per_trade": risk_per_trade / 100,
            "max_open_trades": max_trades,
            "updated_at": datetime.now().isoformat()
        }

        with open(CONFIG_DIR / 'dashboard_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        st.success("Configuración guardada exitosamente!")

# Footer
st.markdown("---")
st.markdown("*Dashboard generado automáticamente - Trading IA System*")