from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import sys

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.optimizer import StrategyOptimizer
from src.backtester import AdvancedBacktester

app = Flask(__name__)

# Global variables for data
optimizer = None
backtester = None
trades_df = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BTC Trading Strategy Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        form { margin: 20px 0; }
        label { display: block; margin: 10px 0 5px; }
        input, select { padding: 8px; width: 200px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        img { max-width: 100%; height: auto; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .filter { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC Trading Strategy Dashboard</h1>
        
        <div class="section">
            <h2>Parameters Input</h2>
            <form action="/optimize" method="post">
                <label>ATR Multiplier:</label>
                <input type="number" step="0.1" name="atr_multiplier" value="1.5" required>
                
                <label>Volume Threshold:</label>
                <input type="number" step="0.1" name="volume_threshold" value="1.2" required>
                
                <label>EMA Short:</label>
                <input type="number" name="ema_short" value="50" required>
                
                <label>EMA Long:</label>
                <input type="number" name="ema_long" value="200" required>
                
                <label>Score Threshold:</label>
                <input type="number" step="0.1" name="score_threshold" value="4.0" required>
                
                <label>Optimization Mode:</label>
                <select name="mode">
                    <option value="bayes">Bayesian Optimization</option>
                    <option value="frontier">Efficient Frontier</option>
                    <option value="sensitivity">Sensitivity Analysis</option>
                </select>
                
                <button type="submit">Run Optimization</button>
            </form>
        </div>
        
        {% if heatmap_img %}
        <div class="section">
            <h2>Optimization Results</h2>
            <img src="data:image/png;base64,{{ heatmap_img }}" alt="Optimization Heatmap">
        </div>
        {% endif %}
        
        {% if trades %}
        <div class="section">
            <h2>Trades Table</h2>
            <div class="filter">
                <label>Filter by Side:</label>
                <select id="side_filter">
                    <option value="">All</option>
                    <option value="long">Long</option>
                    <option value="short">Short</option>
                </select>
                
                <label>Filter by PnL > 0:</label>
                <input type="checkbox" id="pnl_filter">
            </div>
            <table id="trades_table">
                <thead>
                    <tr>
                        <th>Entry Time</th>
                        <th>Side</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>PnL</th>
                        <th>PnL %</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr>
                        <td>{{ trade.entry_time }}</td>
                        <td>{{ trade.side }}</td>
                        <td>{{ "%.2f"|format(trade.entry_price) }}</td>
                        <td>{{ "%.2f"|format(trade.exit_price) }}</td>
                        <td>{{ "%.2f"|format(trade.pnl) }}</td>
                        <td>{{ "%.2f"|format(trade.pnl_pct) }}%</td>
                        <td>{{ trade.score }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
    </div>
    
    <script>
        // Simple client-side filtering
        document.getElementById('side_filter').addEventListener('change', filterTrades);
        document.getElementById('pnl_filter').addEventListener('change', filterTrades);
        
        function filterTrades() {
            const sideFilter = document.getElementById('side_filter').value;
            const pnlFilter = document.getElementById('pnl_filter').checked;
            const rows = document.querySelectorAll('#trades_table tbody tr');
            
            rows.forEach(row => {
                const side = row.cells[1].textContent;
                const pnl = parseFloat(row.cells[4].textContent);
                
                const sideMatch = !sideFilter || side === sideFilter;
                const pnlMatch = !pnlFilter || pnl > 0;
                
                row.style.display = sideMatch && pnlMatch ? '' : 'none';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, heatmap_img=None, trades=None)

@app.route('/optimize', methods=['POST'])
def optimize():
    global optimizer, backtester, trades_df
    
    # Get parameters from form
    params = {
        'atr_multiplier': float(request.form['atr_multiplier']),
        'volume_threshold': float(request.form['volume_threshold']),
        'ema_short': int(request.form['ema_short']),
        'ema_long': int(request.form['ema_long']),
        'score_threshold': float(request.form['score_threshold'])
    }
    
    mode = request.form['mode']
    
    # Load sample data (mock for demo)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    df_5m = pd.DataFrame({
        'Open': 45000 + np.random.randn(1000) * 1000,
        'High': 45500 + np.random.randn(1000) * 1000,
        'Low': 44500 + np.random.randn(1000) * 1000,
        'Close': 45000 + np.random.randn(1000) * 1000,
        'Volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    df_5m['Close'] = df_5m[['Open', 'High', 'Low']].mean(axis=1) + np.random.randn(1000) * 500
    dfs = {'entry': df_5m, 'momentum': df_5m.resample('15min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}),
           'trend': df_5m.resample('1H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})}
    
    # Initialize components
    backtester = AdvancedBacktester()
    optimizer = StrategyOptimizer()
    
    # Run optimization based on mode
    if mode == 'bayes':
        param_bounds = [(1.0, 2.0), (1.0, 2.0), (1.5, 2.5)]  # atr_multi, vol_thresh, tp_rr
        result = optimizer.bayes_opt_sharpe(dfs, param_bounds=param_bounds, n_calls=10)
        best_params = result['best_params']
        heatmap_img = generate_heatmap_from_result(result)
    elif mode == 'frontier':
        param_ranges = {'atr_multi': (1.0, 2.0), 'vol_thresh': (1.0, 2.0), 'tp_rr': (1.5, 2.5)}
        risks, sharpes, param_combinations = optimizer.efficient_frontier_params(dfs, param_ranges, n_points=10)
        best_params = param_combinations[np.argmax(sharpes)] if param_combinations else params
        heatmap_img = generate_frontier_plot_from_data(risks, sharpes)
    else:  # sensitivity
        heatmap_img = generate_heatmap_from_sensitivity(optimizer.param_sensitivity_heatmap(dfs))
        best_params = params
    
    # Run backtest with best params
    result = backtester.run_optimized_backtest(dfs, best_params)
    trades_df = result.get('trades', [])
    
    # Convert trades to list of dicts for template
    trades = []
    if isinstance(trades_df, list):
        for trade in trades_df:
            trades.append({
                'entry_time': str(trade.get('entry_time', '')),
                'side': trade.get('side', ''),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'pnl': trade.get('pnl', 0),
                'pnl_pct': trade.get('pnl_pct', 0) * 100,
                'score': trade.get('score', 0)
            })
    elif hasattr(trades_df, 'iterrows'):
        for _, row in trades_df.iterrows():
            trades.append({
                'entry_time': str(row.get('entry_time', '')),
                'side': row.get('side', ''),
                'entry_price': row.get('entry_price', 0),
                'exit_price': row.get('exit_price', 0),
                'pnl': row.get('pnl', 0),
                'pnl_pct': row.get('pnl_pct', 0) * 100,
                'score': row.get('score', 0)
            })
    
    return render_template_string(HTML_TEMPLATE, heatmap_img=heatmap_img, trades=trades)

def generate_heatmap_from_result(result):
    """Generate simple plot for bayes result"""
    fig, ax = plt.subplots(figsize=(8, 6))
    params = list(result['best_params'].keys())
    values = list(result['best_params'].values())
    ax.bar(params, values)
    ax.set_title(f"Best Parameters (Score: {result['best_score']:.3f})")
    ax.set_ylabel('Value')
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_frontier_plot_from_data(risks, sharpes):
    """Generate frontier plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(risks, sharpes, c='blue', label='Parameter Combinations')
    ax.set_xlabel('Risk (Max Drawdown)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Efficient Frontier')
    ax.legend()
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def generate_heatmap_from_sensitivity(sensitivity_results):
    """Generate heatmap from sensitivity"""
    fig, ax = plt.subplots(figsize=(8, 6))
    # Simple example
    ax.text(0.5, 0.5, 'Sensitivity Analysis Complete', ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)