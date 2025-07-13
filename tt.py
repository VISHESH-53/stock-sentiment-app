import yfinance as yf
data = yf.download("AAPL", start="2015-01-01", end="2024-12-31")
data.to_csv("data/stock_prices_AAPL.csv", index=True)
