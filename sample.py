import yfinance as yf
data = yf.download("TCS.NS", period="5d", progress=False)
print(data.tail())