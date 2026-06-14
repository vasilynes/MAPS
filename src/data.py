import yfinance as yf

data = yf.download('SPY', period='5y', timeout=30)

print(data.head())