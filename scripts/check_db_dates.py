import sqlite3
import pandas as pd

conn = sqlite3.connect("data/trading_data.db")
query = "SELECT symbol, timeframe, MIN(timestamp) as min_date, MAX(timestamp) as max_date, COUNT(*) as count FROM market_data GROUP BY symbol, timeframe"
df = pd.read_sql_query(query, conn)
print(df)
conn.close()
