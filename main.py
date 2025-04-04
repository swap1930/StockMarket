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


# Data loading function with retry and cache
@retry(max_retries=3, delay=1)
@st.cache_data(ttl=60*5)
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date,
            progress=False,
            threads=True
        )
        
        if data.empty:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)
            
        return data.reset_index() if not data.empty else pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
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
    """Display detailed analysis for a given asset"""
    # 1. Validate date range input
    if not hasattr(date_range, '__len__') or len(date_range) != 2:
        st.warning("⚠️ Please select both start and end dates")
        return

    try:
        # 2. Load data with validation
        start_date, end_date = date_range
        data = load_data(ticker, start_date, end_date + timedelta(days=1))
        
        # 3. Verify we got valid data
        if not isinstance(data, pd.DataFrame) or data.empty:
            st.error("🚨 No data found. Please check:")
            st.error(f"- Ticker symbol: {ticker}")
            st.error(f"- Date range: {start_date} to {end_date}")
            return

        # 4. Get asset name safely
        try:
            name = get_asset_name(ticker, asset_type)
        except Exception as e:
            st.warning(f"Couldn't get full asset name: {str(e)}")
            name = ticker

        # 5. Display header
        st.markdown(f"## 📊 {asset_type} Analysis: {name}")

        # --- Rest of your analysis code goes here ---
        # [Keep your existing metric calculations and visualizations]

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.error("Please try different parameters or check the ticker symbol")
        return

   # Replace your metrics section with this more robust version:

    # First ensure we're working with scalar values
    latest_close = data['Close'].iloc[-1] if not data.empty else None
    latest_low = data['Low'].iloc[-1] if not data.empty else None
    latest_high = data['High'].iloc[-1] if not data.empty else None
    latest_volume = data['Volume'].iloc[-1] if not data.empty else None
    
    # Calculate percentage change safely
    change_pct = None
    if len(data) > 1 and not data.empty:
        try:
            change_pct = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / 
                         data['Close'].iloc[-2]) * 100
        except (IndexError, ZeroDivisionError):
            change_pct = None
    
    # Now display the metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Current Price", 
        f"${float(latest_close):.2f}" if latest_close is not None else "N/A", 
        f"{float(change_pct):.2f}%" if change_pct is not None else "N/A"
    )
    col2.metric(
        "Day Range", 
        f"${float(latest_low):.2f} - ${float(latest_high):.2f}" 
        if all(x is not None for x in [latest_low, latest_high]) 
        else "N/A"
    )
    col3.metric(
        "Volume", 
        f"{int(float(latest_volume)):,}" if latest_volume is not None else "N/A"
    )

    
    # 📄 Recent Market Data (Last 15 Entries)
    st.subheader("📄 Recent Market Data (Last 15 Entries)")
    st.dataframe(
        data.sort_values('Date', ascending=False).head(15).reset_index(drop=True),
        use_container_width=True
    )

    # Show the current or predicted data based on the selected date
    if end_date == datetime.today().date():
            st.dataframe(
                data.tail(1).reset_index(drop=True),
                use_container_width=True
            )


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


    # Future Price Prediction Section
    st.subheader("📈 Future Price Prediction")

    # 1. Calculate predictions
    data['Date_Ordinal'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)
    model = LinearRegression()
    model.fit(data[['Date_Ordinal']], data['Close'])

    # Predict for next 30 days
    future_dates = pd.date_range(data['Date'].iloc[-1], periods=30)
    future_ordinals = future_dates.map(pd.Timestamp.toordinal)
    future_preds = model.predict(np.array(future_ordinals).reshape(-1, 1))

    # Convert predictions to float explicitly
    future_preds = future_preds.flatten().astype(float)
    last_price = float(data['Close'].iloc[-1])
    pred_change = float(future_preds[-1]) - last_price
    pred_change_pct = (pred_change / last_price) * 100

    # Projected price metric
    st.metric(
        "Projected Price in 30 Days",
        f"${float(future_preds[-1]):.2f}",
        f"{pred_change_pct:.1f}% ({'↑' if pred_change >= 0 else '↓'} ${abs(pred_change):.2f})",
        delta_color="normal"
    )
    st.info("⚠️ Note: Simple linear projection")

        
    st.subheader("Predicted Market Data")
    future_df = pd.DataFrame({
    'Date': future_dates,
    'Close': future_preds.flatten()
    })
    st.dataframe(future_df)

    # Create tabs for different visualization styles
    tab1, tab2 = st.tabs(["📊 Price Distribution", "📈 Price Trend"])
    with tab1:
        # Title
        st.markdown("**Daily Price Trend**")

        # Date range for predictions
        pred_dates = pd.date_range(start=date_range[0], periods=len(future_preds))

        # Create figure with proper sizing
        pyramid_fig = go.Figure()

        # Interpolate between the start and end prices
        x_start = pred_dates[0]
        x_end = pred_dates[-1]

        # ✅ Make sure these lines are correctly indented
        y_start = float(data['Close'].iloc[-1])
        y_end = float(future_preds[-1])


        # Steps for smoothness
        steps = len(pred_dates)
        x_interp = np.linspace(0, 1, steps)

        # Detect price direction and apply proper curve
        if y_end >= y_start:
            y_interp = y_start + (y_end - y_start) * (0.5 - 0.5 * np.cos(np.pi * x_interp))
            color = '#27AE60'  # Green for upward
        else:
            y_interp = y_start - abs(y_end - y_start) * (0.5 - 0.5 * np.cos(np.pi * x_interp))
            color = '#E74C3C'  # Red for downward

        # Create interpolated x-axis values
        x_curve = pd.date_range(start=x_start, end=x_end, periods=steps)

        # Add the smooth curve line
        pyramid_fig.add_trace(go.Scatter(
            x=x_curve,
            y=y_interp,
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add current price point (blue dot)
        pyramid_fig.add_trace(go.Scatter(
            x=[pred_dates[0]],
            y=[y_start],
            mode='markers+text',
            marker=dict(color='#3498DB', size=12),
            text=["●"],
            textposition="middle center",
            textfont=dict(size=18),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Current Price: ${y_start:.2f}"
        ))

        # Add predicted price point (purple dot)
        pyramid_fig.add_trace(go.Scatter(
            x=[pred_dates[-1]],
            y=[y_end],
            mode='markers+text',
            marker=dict(color='#9B59B6', size=12),
            text=["●"],
            textposition="middle center",
            textfont=dict(size=18),
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Predicted Price: ${y_end:.2f}"
        ))

        # Determine overall line direction
        overall_change = y_end - y_start
        arrow_symbol = "▲" if overall_change >= 0 else "▼"
        arrow_color = '#27AE60' if overall_change >= 0 else '#E74C3C'

        # Add directional arrows
        for i in range(1, len(future_preds)):
            xi = (pred_dates[i] - pred_dates[0]).days
            yi_line = y_start + (y_end - y_start) * (xi / (pred_dates[-1] - pred_dates[0]).days)

            pyramid_fig.add_annotation(
                x=pred_dates[i],
                y=yi_line,
                text=arrow_symbol,
                showarrow=False,
                font=dict(color=arrow_color, size=12)
            )

        # Optimized layout settings to prevent overflow
        pyramid_fig.update_layout(
            template="plotly_white",
            height=400,  # Reduced height for better fit
            margin=dict(l=40, r=40, t=40, b=40),  # Balanced margins
            hovermode="x unified",
            showlegend=False,
            xaxis=dict(
                title="Date",
                tickformat="%b %d",
                rangeslider=dict(visible=False),  # Disable range slider to save space
                automargin=True  # Auto-adjust margins
            ),
            yaxis=dict(
                title="Price ($)",
                range=[min(min(future_preds), y_start) * 0.99,
                       max(max(future_preds), y_start) * 1.01],
                automargin=True
            )
        )

        # Hover formatting
        pyramid_fig.update_traces(
            hovertemplate="<b>%{x|%b %d}</b><br>Price: $%{y:.2f}<extra></extra>"
        )


        # 5. DISPLAY WITH FULL CONTROLS
        st.plotly_chart(
            pyramid_fig,
            use_container_width=True,
            config={
                'modeBarButtonsToRemove': [
                    'select2d',  # Keep this if you want rectangle select
                    'lasso2d',  # Keep this if you want lasso select
                ],
                'displayModeBar': True,
                'displaylogo': False,
                'displayModeBar': 'hover'
            }
        )

        # Legend (unchanged)
        st.markdown("""
                 <div style="text-align: center; margin-top: -10px;">
                     <span style="color: #3498DB; font-weight: bold;">● Current Price</span>
                     <span style="margin: 0 10px;">|</span>
                     <span style="color: #9B59B6; font-weight: bold;">● Predicted Price</span>
                     <span style="margin: 0 10px;">|</span>
                     <span style="color: #27AE60; font-weight: bold;">▲ Increase</span>
                     <span style="margin: 0 10px;">|</span>
                     <span style="color: #E74C3C; font-weight: bold;">▼ Decrease</span>
                 </div>
                 """, unsafe_allow_html=True)

    with tab2:
        # Line Chart - Predicted Prices Over Time
        st.markdown("**Predicted Price Trend**")

        # Check trend direction
        trend_color = '#27AE60' if future_preds[-1] >= future_preds[0] else '#E74C3C'

        # Create line chart
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=future_dates,
            y=future_preds,
            mode='lines+markers',
            line=dict(color=trend_color, width=2),
            marker=dict(size=6),
            name='Predicted Price'
        ))

        # Add reference line for current price
        fig_line.add_hline(
            y=last_price,
            line_dash="dash",
            line_color="#3498DB",
            annotation_text=f"Current Price: ${last_price:.2f}",
            annotation_position="bottom right"
        )

        fig_line.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            showlegend=False,
            template="plotly_white"
        )

        st.plotly_chart(fig_line, use_container_width=True)

# Main app logic

if not ticker:
    if asset_type == "Stock":
        display_stock_overview()
    else:
        display_crypto_overview()
else:
    resolved_ticker = resolve_ticker(ticker, asset_type)

    # Always try to load data first
    start_date, end_date = date_range
    data = load_data(resolved_ticker, start_date, end_date + timedelta(days=1))

    if data.empty:
        # Only show error if we can't get data
        if asset_type == "Stock":
            example_tickers = ['AAPL', 'RELIANCE.NS', '005930.KS']
            example_names = ['Apple', 'Reliance', 'Samsung']
            st.error(
                f"⚠️ Could not find data for '{ticker}'. Valid examples: "
                f"{', '.join(example_tickers)} or {', '.join(example_names)}"
            )
        else:
            example_tickers = ['BTC-USD', 'SOL-USD']
            example_names = ['Bitcoin', 'Solana']
            st.error(
                f"⚠️ Could not find data for '{ticker}'. Valid examples: "
                f"{', '.join(example_tickers)} or {', '.join(example_names)}"
            )
    else:
        display_asset_analysis(resolved_ticker, asset_type)

