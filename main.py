import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from functools import wraps
import time
import random
import requests
import urllib3
from packaging import version

# Modern package version check import
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # Python < 3.8

# Initialize yfinance with proper settings
# Initialize yfinance with proper settings
try:
    # Try the new recommended initialization first
    yf.set_option("use_pydantic", False)
except AttributeError:
    # Fallback to older initialization if needed
    try:
        yf.pdr_override()  # Only for older yfinance versions
    except AttributeError:
        pass  # Skip if neither method is available
        
# Configure requests session for yfinance
urllib3.disable_warnings()

yf.pdr_override()  # Only needed if using pandas_datareader compatibility

# Set yfinance options
yf.set_option("use_pydantic", False)

# Configure Streamlit page
st.set_page_config(page_title="MarketMatrix", layout="wide", page_icon="🌐")
st.title("🌐 MarketMatrix")

# Hide default Streamlit elements
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

def check_versions():
    required = {
        'yfinance': '0.2.38',
        'pandas': '2.0.0',
        'requests': '2.31.0'
    }
    for pkg, ver in required.items():
        try:
            current = importlib_metadata.version(pkg)
            if version.parse(current) < version.parse(ver):
                st.warning(f"⚠️ {pkg} version {current} is below recommended {ver}")
        except Exception as e:
            st.error(f"Error checking {pkg} version: {str(e)}")

check_versions()

# Retry decorator for API calls
def retry(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise e
                    time.sleep(delay * (1 + random.random()))
        return wrapper
    return decorator

# Asset examples
STOCK_EXAMPLES = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 
    'AMZN': 'Amazon', 'TSLA': 'Tesla', 'META': 'Meta', 
    'NVDA': 'NVIDIA', 'JPM': 'JPMorgan', 'RELIANCE.NS': 'Reliance',
    'TCS.NS': 'TCS', 'HDFCBANK.NS': 'HDFC Bank', 'INFY.NS': 'Infosys',
    'BHARTIARTL.NS': 'Airtel', '005930.KS': 'Samsung (KR)',
    '7203.T': 'Toyota (JP)', 'HSBA.L': 'HSBC (UK)'
}

CRYPTO_EXAMPLES = {
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple', 'SOL-USD': 'Solana', 'DOGE-USD': 'Dogecoin',
    'ADA-USD': 'Cardano', 'DOT-USD': 'Polkadot', 'SHIB-USD': 'Shiba Inu',
    'MATIC-USD': 'Polygon'
}


# Sidebar inputs
asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])
st.sidebar.markdown("**Examples**: ")

# Fixed the syntax error in this line - removed the extra list() and fixed parentheses
example_items = list(STOCK_EXAMPLES.items())[:5] if asset_type == "Stock" else list(CRYPTO_EXAMPLES.items())[:5]
for ticker, desc in example_items:
    st.sidebar.markdown(f"• {ticker} ({desc})")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "")
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
date_range = st.sidebar.date_input("Select Date Range", [start_date, end_date])

# Connection test function
def check_connection():
    try:
        response = session.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d", timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False

if st.sidebar.checkbox("Run connection test"):
    if check_connection():
        st.sidebar.success("✅ Connection to Yahoo Finance successful")
    else:
        st.sidebar.error("❌ Could not connect to Yahoo Finance")

# Data loading function with retry and cache
@retry(max_retries=3, delay=1)
@st.cache_data(ttl=60*5)
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date + timedelta(days=1),
            progress=False,
            threads=True
        )
        if data.empty:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date + timedelta(days=1))
            
        if not data.empty:
            data.reset_index(inplace=True)
            return data
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Ticker resolution functions
def resolve_ticker(input_str, asset_type):
    input_str = input_str.strip().upper()
    if not input_str:
        return input_str

    if asset_type == "Stock":
        if input_str in STOCK_EXAMPLES:
            return input_str
        if input_str + '.NS' in STOCK_EXAMPLES:
            return input_str + '.NS'

        input_lower = input_str.lower()
        for t, name in STOCK_EXAMPLES.items():
            if input_lower == name.lower():
                return t
            if input_lower in name.lower():
                return t

        if not input_str.endswith(('.NS', '.KS', '.T', '.L')):
            for suffix in ['.NS', '.KS', '.T', '.L']:
                if input_str + suffix in STOCK_EXAMPLES:
                    return input_str + suffix
    else:
        if input_str in CRYPTO_EXAMPLES:
            return input_str
        if input_str + '-USD' in CRYPTO_EXAMPLES:
            return input_str + '-USD'
        for t, name in CRYPTO_EXAMPLES.items():
            if input_str.lower() == name.lower():
                return t

    return input_str

@st.cache_data
def get_asset_name(ticker, asset_type):
    try:
        if asset_type == "Stock":
            stock = yf.Ticker(ticker)
            return stock.info.get("longName", ticker)
        return ticker
    except Exception:
        return ticker

# Dashboard display functions
def display_stock_overview():
    st.subheader("📈 Stock Market Dashboard")
    
    # Major indices
    st.markdown("### Global Indices")
    indices = {
        '^GSPC': ('S&P 500', '$'), '^IXIC': ('NASDAQ', '$'), 
        '^DJI': ('Dow Jones', '$'), '^NSEI': ('Nifty 50', '₹'),
        '^HSI': ('Hang Seng', 'HK$'), '^FTSE': ('FTSE 100', '£'),
        '^GDAXI': ('DAX', '€')
    }
    
    cols = st.columns(len(indices))
    for i, (ticker, (name, symbol)) in enumerate(indices.items()):
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) > 1:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2]
                change = ((current - previous) / previous * 100) if previous != 0 else 0
                cols[i].metric(name, f"{symbol}{current:,.2f}", f"{change:.2f}%")
            else:
                current = data['Close'].iloc[-1] if not data.empty else 0
                cols[i].metric(name, f"{symbol}{current:,.2f}", "N/A")
        except Exception:
            cols[i].error(f"Error loading {name}")

    # Popular stocks
    st.markdown("### Popular Stocks")
    stock_data = {}
    for ticker, name in STOCK_EXAMPLES.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty and len(hist) > 1:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = ((current - previous) / previous * 100) if previous != 0 else 0
                symbol = '$'
                if '.NS' in ticker: symbol = '₹'
                elif '.KS' in ticker: symbol = '₩'
                elif '.T' in ticker: symbol = '¥'
                
                stock_data[ticker] = {
                    'name': name, 'price': current, 'change': change,
                    'volume': hist['Volume'].iloc[-1], 'symbol': symbol
                }
        except Exception:
            pass

    cols = st.columns(4)
    for i, (ticker, data) in enumerate(stock_data.items()):
        with cols[i % 4]:
            st.metric(
                f"{data['name']} ({ticker})",
                f"{data['symbol']}{data['price']:,.2f}",
                f"{data['change']:.2f}%"
            )
            st.caption(f"Vol: {data['volume']:,}")

    if st.button("🔄 Refresh Stock Data"):
        st.cache_data.clear()

def display_crypto_overview():
    st.subheader("💰 Cryptocurrency Dashboard")
    st.markdown("### Top Cryptocurrencies")
    
    crypto_data = {}
    cols = st.columns(min(5, len(CRYPTO_EXAMPLES)))
    for i, (ticker, name) in enumerate(CRYPTO_EXAMPLES.items()):
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty and len(hist) > 1:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2]
                change = ((current - previous) / previous * 100) if previous != 0 else 0
                cols[i % 5].metric(name, f"${current:,.2f}", f"{change:.2f}%")
                cols[i % 5].caption(f"24h Vol: ${hist['Volume'].iloc[-1]:,}")
                crypto_data[ticker] = {'name': name, 'data': hist}
        except Exception:
            pass

    st.markdown("### Price Trends (24h)")
    if crypto_data:
        fig = go.Figure()
        for ticker, data in crypto_data.items():
            fig.add_trace(go.Scatter(
                x=data['data'].index, y=data['data']['Close'],
                name=data['name'], line=dict(width=1.5)
            ))
        fig.update_layout(
            height=500, xaxis_title="Time", yaxis_title="Price (USD)",
            template="plotly_dark", hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Refresh Crypto Data"):
        st.cache_data.clear()

def display_asset_analysis(ticker, asset_type):
    if len(date_range) != 2:
        st.warning("Please select both start and end dates")
        return

    start_date, end_date = date_range
    data = load_data(ticker, start_date, end_date + timedelta(days=1))

    if data.empty:
        st.error("No data found. Please check the ticker symbol.")
        return

    # Basic metrics
    name = get_asset_name(ticker, asset_type)
    st.markdown(f"## 📊 {asset_type} Analysis: {name}")

    latest_close = data.iloc[-1]['Close']
    latest_low = data.iloc[-1]['Low']
    latest_high = data.iloc[-1]['High']
    latest_volume = data.iloc[-1]['Volume']
    
    if len(data) > 1:
        change_pct = ((latest_close - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) 
    else:
        change_pct = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${latest_close:.2f}", f"{change_pct:.2f}%")
    col2.metric("Day Range", f"${latest_low:.2f} - ${latest_high:.2f}")
    col3.metric("Volume", f"{latest_volume:,}")

    # Recent data
    st.subheader("📄 Recent Market Data")
    st.dataframe(data.sort_values('Date', ascending=False).head(15), use_container_width=True)

    if end_date == datetime.today().date():
        st.subheader("💹 Current Market Data")
        st.dataframe(data.tail(1))

    # Daily returns analysis
    st.subheader("📉 Daily Return Analysis")
    data['Daily Return'] = data['Close'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    with col1:
        hist_fig = go.Figure(go.Histogram(
            x=data['Daily Return'].dropna(), nbinsx=50,
            marker_color='#3498DB', opacity=0.7
        ))
        hist_fig.update_layout(height=400, xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with col2:
        line_fig = go.Figure(go.Scatter(
            x=data['Date'], y=data['Daily Return'],
            line=dict(color='#27AE60', width=1)
        ))
        line_fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Return (%)")
        st.plotly_chart(line_fig, use_container_width=True)

    # Moving averages
    st.subheader("Moving Averages Analysis")
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Histogram(x=data['MA20'].dropna(), name='MA20', marker_color='#F39C12'))
    ma_fig.add_trace(go.Histogram(x=data['MA50'].dropna(), name='MA50', marker_color='#27AE60'))
    ma_fig.update_layout(barmode='overlay', xaxis_title='Moving Average Value', yaxis_title='Frequency')
    st.plotly_chart(ma_fig, use_container_width=True)

    # Future price prediction
    st.subheader("📈 Future Price Prediction")
    if len(data) > 1:
        data['Date_Ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
        model = LinearRegression()
        model.fit(data[['Date_Ordinal']], data['Close'])
        
        future_dates = pd.date_range(data['Date'].iloc[-1], periods=30)
        future_ordinals = future_dates.map(pd.Timestamp.toordinal)
        future_preds = model.predict(np.array(future_ordinals).reshape(-1, 1))
        
        last_price = data['Close'].iloc[-1]
        pred_change = future_preds[-1] - last_price
        pred_change_pct = (pred_change / last_price) * 100
        
        st.metric(
            "Projected Price in 30 Days",
            f"${future_preds[-1]:.2f}",
            f"{pred_change_pct:.1f}% ({'↑' if pred_change >= 0 else '↓'} ${abs(pred_change):.2f})"
        )
        st.info("⚠️ Note: Simple linear projection")

        # Visualization tabs
        tab1, tab2 = st.tabs(["📊 Price Distribution", "📈 Price Trend"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_preds.flatten(),
                mode='lines+markers', line=dict(width=2),
                marker=dict(size=6), name='Predicted Price'
            ))
            fig.add_hline(
                y=last_price, line_dash="dash",
                annotation_text=f"Current: ${last_price:.2f}"
            )
            fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.dataframe(pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_preds.flatten()
            }))

# Main app logic
if not ticker:
    if asset_type == "Stock":
        display_stock_overview()
    else:
        display_crypto_overview()
else:
    resolved_ticker = resolve_ticker(ticker, asset_type)
    data = load_data(resolved_ticker, date_range[0], date_range[1] + timedelta(days=1))
    
    if data.empty:
        examples = (STOCK_EXAMPLES if asset_type == "Stock" else CRYPTO_EXAMPLES)
        st.error(f"⚠️ Could not find data for '{ticker}'. Try: {', '.join(list(examples.keys())[:3])}")
    else:
        display_asset_analysis(resolved_ticker, asset_type)
