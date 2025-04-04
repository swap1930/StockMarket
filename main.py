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

import yfinance as yf
yf.pdr_override()  # Fix for some yfinance issues

# Set user agent to prevent blocking
import requests
import urllib3
urllib3.disable_warnings()
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})
yf.set_session(session)

# Configure page
st.set_page_config(page_title=" MarketMatrix", layout="wide",
                   page_icon="🌐")
st.title("🌐 MarketMatrix ")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Try hiding the toolbar via its role attribute */
    [data-testid="stToolbar"] {display: none !important;}
    </style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar input
asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])

STOCK_EXAMPLES = {
    # US Stocks
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'TSLA': 'Tesla',
    'META': 'Meta',
    'NVDA': 'NVIDIA',
    'JPM': 'JPMorgan',

    # Indian Stocks
    'RELIANCE.NS': 'Reliance',
    'TCS.NS': 'TCS',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'BHARTIARTL.NS': 'Airtel',

    # International
    '005930.KS': 'Samsung (KR)',
    '7203.T': 'Toyota (JP)',
    'HSBA.L': 'HSBC (UK)'
}

CRYPTO_EXAMPLES = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple',
    'SOL-USD': 'Solana',
    'DOGE-USD': 'Dogecoin',
    'ADA-USD': 'Cardano',
    'DOT-USD': 'Polkadot',
    'SHIB-USD': 'Shiba Inu',
    'MATIC-USD': 'Polygon'
}
# Show appropriate examples (limited to 5)
st.sidebar.markdown("**Examples**: ")
if asset_type == "Stock":
    # Get first 5 stock examples
    for ticker, desc in list(STOCK_EXAMPLES.items())[:5]:
        st.sidebar.markdown(f"• {ticker} ({desc})")
else:
    # Get first 5 crypto examples
    for ticker, desc in list(CRYPTO_EXAMPLES.items())[:5]:
        st.sidebar.markdown(f"• {ticker} ({desc})")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "")

# Date range selector
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
date_range = st.sidebar.date_input("Select Date Range", [start_date, end_date])

def check_versions():
    import pkg_resources
    required = {
        'yfinance': '0.2.38',
        'pandas': '2.0.0',
        'requests': '2.31.0'
    }
    for pkg, ver in required.items():
        current = pkg_resources.get_distribution(pkg).version
        if pkg_resources.parse_version(current) < pkg_resources.parse_version(ver):
            st.warning(f"⚠️ {pkg} version {current} is below recommended {ver}")

# Call this early in your app
check_versions()
@st.cache_data
def get_asset_name(ticker, asset_type):
    try:
        if asset_type == "Stock":
            stock = yf.Ticker(ticker)
            return stock.info.get("longName", ticker)
        else:
            return ticker
    except Exception as e:
        st.error(f"Error fetching asset name: {str(e)}")
        return ticker

def check_versions():
    import pkg_resources
    required = {
        'yfinance': '0.2.38',
        'pandas': '2.0.0',
        'requests': '2.31.0'
    }
    for pkg, ver in required.items():
        current = pkg_resources.get_distribution(pkg).version
        if pkg_resources.parse_version(current) < pkg_resources.parse_version(ver):
            st.warning(f"⚠️ {pkg} version {current} is below recommended {ver}")

# Call this early in your app
check_versions()


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
            st.warning(f"No data found for {ticker}. Trying again with different parameters...")
            # Try alternative approach
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date + timedelta(days=1))
            
        if data.empty:
            return pd.DataFrame()
            
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()


def validate_ticker(ticker, asset_type):
    """More flexible validation that first checks our examples, then tries yfinance"""
    if not ticker:
        return False

    # First check against our known examples
    resolved = resolve_ticker(ticker, asset_type)
    if asset_type == "Stock":
        if resolved in STOCK_EXAMPLES or (resolved + '.NS') in STOCK_EXAMPLES:
            return True
    else:
        if resolved in CRYPTO_EXAMPLES or (resolved + '-USD') in CRYPTO_EXAMPLES:
            return True

    # If not in our examples, still allow it (yfinance might know it)
    return True




def resolve_ticker(input_str, asset_type):
    """Resolve input to best ticker format"""
    input_str = input_str.strip().upper()
    if not input_str:
        return input_str

    # For stocks
    if asset_type == "Stock":
        # Check exact matches first
        if input_str in STOCK_EXAMPLES:
            return input_str
        if input_str + '.NS' in STOCK_EXAMPLES:
            return input_str + '.NS'

        # Check name matches (case insensitive)
        input_lower = input_str.lower()
        for ticker, name in STOCK_EXAMPLES.items():
            if input_lower == name.lower():
                return ticker
            if input_lower in name.lower():  # Partial match
                return ticker

        # For international stocks, try common suffixes
        if not input_str.endswith(('.NS', '.KS', '.T', '.L')):
            test_tickers = [
                input_str + '.NS',  # Indian stocks
                input_str + '.KS',  # Korean stocks
                input_str + '.T',  # Japanese stocks
                input_str + '.L'  # London stocks
            ]
            for test_ticker in test_tickers:
                if test_ticker in STOCK_EXAMPLES:
                    return test_ticker

    # For cryptos
    else:
        if input_str in CRYPTO_EXAMPLES:
            return input_str
        if input_str + '-USD' in CRYPTO_EXAMPLES:
            return input_str + '-USD'
        for ticker, name in CRYPTO_EXAMPLES.items():
            if input_str.lower() == name.lower():
                return ticker

    # Return original if no match found (will attempt with yfinance)
    return input_str


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

# Then decorate your functions
@retry(max_retries=3, delay=1)
@st.cache_data(ttl=60*5)
def load_data(ticker, start_date, end_date):
    # ... existing code ...
  

def check_connection():
    try:
        response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=1d")
        st.write(f"Connection test status: {response.status_code}")
        st.write(f"Response content: {response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Connection test failed: {str(e)}")
        return False

# Call this somewhere in your app
if st.sidebar.checkbox("Run connection test"):
    check_connection()

def display_stock_overview():
    st.subheader("📈 Stock Market Dashboard")

    # Major indices
    st.markdown("### Global Indices")
    indices = {
        '^GSPC': ('S&P 500', '$'),
        '^IXIC': ('NASDAQ', '$'),
        '^DJI': ('Dow Jones', '$'),
        '^NSEI': ('Nifty 50', '₹'),
        '^HSI': ('Hang Seng', 'HK$'),
        '^FTSE': ('FTSE 100', '£'),
        '^GDAXI': ('DAX', '€')
    }

    # Get index data
    cols = st.columns(len(indices))
    for i, (ticker, (name, symbol)) in enumerate(indices.items()):
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if not data.empty and len(data) > 1:
                current = float(data['Close'].iloc[-1])
                previous = float(data['Close'].iloc[-2])
                change = ((current - previous) / previous * 100) if previous != 0 else 0

                cols[i].metric(
                    name,
                    f"{symbol}{current:,.2f}",
                    f"{change:.2f}%",
                    delta_color="normal"  # always use "normal"
                )

            else:
                current = float(data['Close'].iloc[-1]) if not data.empty else 0
                cols[i].metric(name, f"{symbol}{current:,.2f}", "N/A")
        except Exception as e:
            cols[i].error(f"Error loading {name}")

    # Popular stocks
    st.markdown("### Popular Stocks")
    stock_data = {}
    for ticker, name in STOCK_EXAMPLES.items():
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty and len(hist) > 1:
                current = float(hist['Close'].iloc[-1])
                previous = float(hist['Close'].iloc[-2])
                change = ((current - previous) / previous * 100) if previous != 0 else 0

                # Determine currency symbol
                symbol = '$'
                if '.NS' in ticker:
                    symbol = '₹'
                elif '.KS' in ticker:
                    symbol = '₩'
                elif '.T' in ticker:
                    symbol = '¥'

                stock_data[ticker] = {
                    'name': name,
                    'price': current,
                    'change': change,
                    'volume': int(hist['Volume'].iloc[-1]),
                    'symbol': symbol
                }
        except Exception as e:
            st.error(f"Error loading {ticker}: {str(e)}")

    # Display in grid
    cols = st.columns(4)
    for i, (ticker, data) in enumerate(stock_data.items()):
        with cols[i % 4]:
            st.metric(
                f"{data['name']} ({ticker})",
                f"{data['symbol']}{data['price']:,.2f}",
                f"{data['change']:.2f}%",
                delta_color="normal"  # always use "normal"
            )

            st.caption(f"Vol: {data['volume']:,}")

    if st.button("🔄 Refresh Stock Data", key="refresh_stock"):
        st.cache_data.clear()
        st.rerun()




def display_crypto_overview():
    st.subheader("💰 Cryptocurrency Dashboard")

    # Top cryptos
    st.markdown("### Top Cryptocurrencies")
    crypto_data = {}

    cols = st.columns(min(5, len(CRYPTO_EXAMPLES)))
    for i, (ticker, name) in enumerate(CRYPTO_EXAMPLES.items()):
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if not hist.empty and len(hist) > 1:
                current = float(hist['Close'].iloc[-1])
                previous = float(hist['Close'].iloc[-2])
                change = ((current - previous) / previous * 100) if previous != 0 else 0

                cols[i % 5].metric(
                    name,
                    f"${current:,.2f}",
                    f"{change:.2f}%",
                    delta_color="normal"
                )
                cols[i % 5].caption(f"24h Vol: ${int(hist['Volume'].iloc[-1]):,}")

                crypto_data[ticker] = {'name': name, 'data': hist}
        except Exception as e:
            st.error(f"Error loading {ticker}: {str(e)}")

    # Price chart
    st.markdown("### Price Trends (24h)")
    if crypto_data:
        fig = go.Figure()
        for ticker, data in crypto_data.items():
            fig.add_trace(go.Scatter(
                x=data['data'].index,
                y=data['data']['Close'],
                name=data['name'],
                mode='lines+markers',
                line=dict(width=1.5)
            ))

        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 Refresh Crypto Data", key="refresh_crypto"):
        st.cache_data.clear()
        st.rerun()




def display_asset_analysis(ticker, asset_type):
    if len(date_range) != 2:
        st.warning("Please select both start and end dates")
        return

    start_date, end_date = date_range
    data = load_data(ticker, start_date, end_date + timedelta(days=1))

    if data.empty:
        st.error("No data found. Please check the ticker symbol.")
        return

    # Initialize future prediction variables with default values
    future_dates = pd.Series(dtype='datetime64[ns]')
    future_preds = np.array([])
    last_price = 0.0

    # Only calculate predictions if we have enough data
    if len(data) > 1:
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

    # Asset header
    name = get_asset_name(ticker, asset_type)
    st.markdown(f"## 📊 {asset_type} Analysis: {name}")


    # Get scalar values for metrics
    latest_close = float(data.iloc[-1]['Close'])
    latest_low = float(data.iloc[-1]['Low'])
    latest_high = float(data.iloc[-1]['High'])
    latest_volume = int(data.iloc[-1]['Volume'])

    # Calculate percentage change safely
    if len(data) > 1:
        prev_close = float(data.iloc[-2]['Close'])
        change_pct = ((latest_close - prev_close) / prev_close * 100) if prev_close != 0 else 0
    else:
        change_pct = 0
        prev_close = latest_close

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price",
            f"${latest_close:.2f}",
            f"{change_pct:.2f}%",
            delta_color="normal"
        )
    with col2:
        st.metric(
            "Day Range",
            f"${latest_low:.2f} - ${latest_high:.2f}"
        )
    with col3:
        st.metric("Volume", f"{latest_volume:,}")




    # 📄 Recent Market Data (Last 15 Entries)
    st.subheader("📄 Recent Market Data (Last 15 Entries)")
    st.dataframe(
        data.sort_values('Date', ascending=False).head(15).reset_index(drop=True),
        use_container_width=True
    )

    # Show the current or predicted data based on the selected date
    if end_date == datetime.today().date():
        st.subheader("💹 Current Market Data")
        st.dataframe(data.tail(1))  # Display the latest data row




    # 📉 Daily Returns with better visualization
    st.subheader("📉 Daily Return Analysis")
    data['Daily Return'] = data['Close'].pct_change() * 100  # Convert to percentage

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("*Daily Return Distribution*")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=data['Daily Return'].dropna(),
            nbinsx=50,
            marker_color='#3498DB',
            opacity=0.7
        ))
        hist_fig.update_layout(
            height=400,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            bargap=0.1
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.markdown("*Daily Returns Over Time*")
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Daily Return'],
            mode='lines',
            line=dict(color='#27AE60', width=1),
            name='Daily Return'
        ))
        line_fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Return (%)",
            showlegend=False
        )
        st.plotly_chart(line_fig, use_container_width=True)

    # Moving Averages Section (Updated to match style)
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    st.subheader("Moving Averages Analysis")

    # Create two columns for side-by-side charts



    # Moving Averages Distribution
    st.markdown("**Moving Averages Distribution**")
    ma_dist_fig = go.Figure()
    ma_dist_fig.add_trace(go.Histogram(
        x=data['MA20'].dropna(),
        name='MA20',
        marker_color='#F39C12',
        opacity=0.7
    ))
    ma_dist_fig.add_trace(go.Histogram(
        x=data['MA50'].dropna(),
        name='MA50',
        marker_color='#27AE60',
        opacity=0.7
    ))
    ma_dist_fig.update_layout(
        barmode='overlay',
        xaxis_title='Moving Average Value',
        yaxis_title='Frequency',
        template="plotly_white"
    )
    st.plotly_chart(ma_dist_fig, use_container_width=True)




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
        y_start = latest_close
        y_end = future_preds[-1]

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

