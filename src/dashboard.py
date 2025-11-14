"""
Streamlit Dashboard for BTC IFVG Trading System
================================================
Visualización interactiva de:
- Resultados de backtesting
- Equity curve
- Tabla de trades
- Gráficos de precio con señales IFVG
- Volume Profile heatmap
- Métricas de performance
"""

from src.indicators import calculate_all_indicators
from src.data_fetcher import DataFetcher
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# Page configuration
st.set_page_config(
    page_title="BTC IFVG Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #2ca02c;
    }
    .negative {
        color: #d62728;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_backtest_results():
    """Carga resultados de backtest más recientes"""
    results_dir = Path('results')

    if not results_dir.exists():
        return None, None, None

    # Find most recent files
    trades_files = sorted(results_dir.glob('backtest_trades_*.csv'))
    equity_files = sorted(results_dir.glob('backtest_equity_*.csv'))
    metrics_files = sorted(results_dir.glob('backtest_metrics_*.json'))

    if not trades_files or not equity_files or not metrics_files:
        return None, None, None

    # Load most recent
    trades_df = pd.read_csv(trades_files[-1])
    equity_df = pd.read_csv(equity_files[-1])

    with open(metrics_files[-1], 'r') as f:
        metrics = json.load(f)

    return trades_df, equity_df, metrics


@st.cache_data(ttl=300)
def load_paper_trading_history():
    """Carga historial de paper trading"""
    trades_file = Path('logs') / 'paper_trades.json'

    if not trades_file.exists():
        return None

    try:
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        return pd.DataFrame(trades)
    except Exception as e:
        st.error(f"Error loading paper trades: {e}")
        return None


@st.cache_data(ttl=300)
def load_market_data(symbol='BTCUSD', timeframe='5Min', days_back=30):
    """Carga datos de mercado con indicadores"""
    try:
        fetcher = DataFetcher()
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days_back)

        df = fetcher.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and len(df) > 0:
            # Calculate indicators
            df = calculate_all_indicators(df)
            return df

        return None
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None


def plot_equity_curve(equity_df, initial_capital):
    """Gráfico de equity curve"""
    fig = go.Figure()

    # Equity line
    fig.add_trace(go.Scatter(
        x=list(range(len(equity_df))),
        y=equity_df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))

    # Initial capital line
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital"
    )

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade Number",
        yaxis_title="Equity ($)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_trades_distribution(trades_df):
    """Distribución de P&L de trades"""
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=trades_df['pnl'],
        nbinsx=30,
        name='P&L Distribution',
        marker_color='#1f77b4',
        opacity=0.7
    ))

    # Add mean line
    mean_pnl = trades_df['pnl'].mean()
    fig.add_vline(
        x=mean_pnl,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ${mean_pnl:.2f}"
    )

    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )

    return fig


def plot_candlestick_with_signals(df, max_bars=500):
    """Candlestick chart con señales IFVG"""
    # Limit data for performance
    df = df.iloc[-max_bars:].copy()

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Signals', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#2ca02c',
            decreasing_line_color='#d62728'
        ),
        row=1, col=1
    )

    # Add EMAs
    if 'EMA20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA20'],
                mode='lines',
                name='EMA20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

    if 'EMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA50'],
                mode='lines',
                name='EMA50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )

    # Add signals
    buy_signals = df[df['signal'] > 0]
    sell_signals = df[df['signal'] < 0]

    if len(buy_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Low'] * 0.995,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(color='darkgreen', width=1)
                )
            ),
            row=1, col=1
        )

    if len(sell_signals) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['High'] * 1.005,
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(color='darkred', width=1)
                )
            ),
            row=1, col=1
        )

    # Volume
    colors = ['green' if row['Close'] >= row['Open'] else 'red'
              for _, row in df.iterrows()]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="BTC/USD Price Chart with IFVG Signals",
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=700,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_drawdown(equity_df):
    """Gráfico de drawdown"""
    equity = equity_df['equity'].values
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len(drawdown))),
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red', width=2),
        fillcolor='rgba(214, 39, 40, 0.2)'
    ))

    fig.update_layout(
        title="Equity Drawdown",
        xaxis_title="Trade Number",
        yaxis_title="Drawdown (%)",
        template='plotly_white',
        height=300
    )

    return fig


def display_metrics(metrics):
    """Muestra métricas en cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Return",
            f"{metrics.get('total_return', 0):.2f}%",
            delta=None
        )
        st.metric(
            "Win Rate",
            f"{metrics.get('win_rate', 0):.1f}%"
        )

    with col2:
        st.metric(
            "Profit Factor",
            f"{metrics.get('profit_factor', 0):.2f}"
        )
        st.metric(
            "Total Trades",
            f"{metrics.get('total_trades', 0)}"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        )
        st.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.2f}%"
        )

    with col4:
        st.metric(
            "Calmar Ratio",
            f"{metrics.get('calmar_ratio', 0):.2f}"
        )
        st.metric(
            "Avg Risk/Reward",
            f"{metrics.get('avg_rr', 0):.2f}"
        )


def display_trades_table(trades_df):
    """Muestra tabla de trades"""
    # Format columns
    display_df = trades_df.copy()

    # Format datetime columns
    if 'entry_time' in display_df.columns:
        display_df['entry_time'] = pd.to_datetime(
            display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    if 'exit_time' in display_df.columns:
        display_df['exit_time'] = pd.to_datetime(
            display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')

    # Format numeric columns
    if 'pnl' in display_df.columns:
        display_df['pnl'] = display_df['pnl'].round(2)
    if 'pnl_percent' in display_df.columns:
        display_df['pnl_percent'] = display_df['pnl_percent'].round(2)

    # Select columns to display
    cols_to_display = ['entry_time', 'exit_time', 'direction', 'entry_price',
                       'exit_price', 'pnl', 'pnl_percent', 'exit_reason']

    available_cols = [col for col in cols_to_display if col in display_df.columns]

    # Style the dataframe
    def color_pnl(val):
        if isinstance(val, (int, float)):
            return 'color: green' if val > 0 else 'color: red'
        return ''

    styled_df = display_df[available_cols].style.applymap(
        color_pnl,
        subset=['pnl', 'pnl_percent'] if 'pnl' in display_df.columns else []
    )

    st.dataframe(styled_df, use_container_width=True, height=400)


def main():
    """Main dashboard function"""

    # Header
    st.markdown('<h1 class="main-header">📊 BTC IFVG Trading Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Mode selection
        mode = st.selectbox(
            "Dashboard Mode",
            ["Backtest Results", "Paper Trading", "Live Analysis"],
            index=0
        )

        st.divider()

        # Data refresh
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Info
        st.info("""
        **IFVG Strategy Dashboard**

        Visualiza resultados de:
        - Backtesting histórico
        - Paper trading en vivo
        - Análisis de mercado

        Indicadores:
        - IFVG (Fair Value Gaps)
        - Volume Profile
        - EMAs (20/50/100/200)
        """)

    # Main content based on mode
    if mode == "Backtest Results":
        st.header("📈 Backtest Results")

        # Load data
        trades_df, equity_df, metrics = load_backtest_results()

        if trades_df is None:
            st.warning("⚠️ No backtest results found. Run a backtest first.")
            st.code("python main.py --mode backtest --start 2023-01-01 --end 2025-11-12")
            return

        # Display metrics
        st.subheader("Performance Metrics")
        display_metrics(metrics)

        st.divider()

        # Charts
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Equity Curve", "📉 Drawdown", "📊 P&L Distribution", "📋 Trades"])

        with tab1:
            initial_capital = metrics.get('final_capital', 10000) if metrics else 10000
            st.plotly_chart(
                plot_equity_curve(equity_df, initial_capital),
                use_container_width=True
            )

        with tab2:
            st.plotly_chart(
                plot_drawdown(equity_df),
                use_container_width=True
            )

        with tab3:
            st.plotly_chart(
                plot_trades_distribution(trades_df),
                use_container_width=True
            )

        with tab4:
            st.subheader(f"All Trades ({len(trades_df)} total)")
            display_trades_table(trades_df)

    elif mode == "Paper Trading":
        st.header("🤖 Paper Trading Monitor")

        # Load paper trading data
        trades_df = load_paper_trading_history()

        if trades_df is None or len(trades_df) == 0:
            st.warning("⚠️ No paper trading history found. Start paper trading first.")
            st.code("python main.py --mode paper --symbol BTC/USD --capital 10000")
            return

        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        win_rate = (len(trades_df[trades_df['pnl'] > 0]) /
                    len(trades_df) * 100) if len(trades_df) > 0 else 0

        # Display summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            st.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{total_pnl:.2f}")
        with col3:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            avg_pnl = trades_df['pnl'].mean()
            st.metric("Avg P&L", f"${avg_pnl:.2f}")

        st.divider()

        # Trades table
        st.subheader("Recent Trades")
        display_trades_table(trades_df.tail(50))

    elif mode == "Live Analysis":
        st.header("📊 Live Market Analysis")

        # Parameters
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input("Symbol", value="BTCUSD")
        with col2:
            timeframe = st.selectbox("Timeframe", ["5Min", "15Min", "1H"], index=0)

        # Load market data
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading market data..."):
                df = load_market_data(symbol, timeframe, days_back=30)

                if df is not None:
                    st.success(f"✅ Loaded {len(df)} bars")

                    # Display chart
                    st.plotly_chart(
                        plot_candlestick_with_signals(df),
                        use_container_width=True
                    )

                    # Current signals
                    st.subheader("Recent Signals")
                    signals_df = df[df['signal'] != 0].tail(10)

                    if len(signals_df) > 0:
                        st.dataframe(
                            signals_df[['signal', 'confidence', 'Close', 'Volume']],
                            use_container_width=True
                        )
                    else:
                        st.info("No recent signals")
                else:
                    st.error("❌ Failed to load data")


if __name__ == "__main__":
    main()
