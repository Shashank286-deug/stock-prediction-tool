import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IndusVC AI Terminal", layout="wide", page_icon="🚀")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30333d;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #00D26A;
        color: black;
        font-weight: bold;
    }
    /* Enhance the Analyst Report Box */
    .stInfo {
        background-color: #0e1117;
        border: 1px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION: SECTOR LISTS ---
SECTORS = {
    "TECH & AI": ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'GOOG', 'AMZN', 'META'],
    "CRYPTO": ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD'],
    "FINANCE": ['JPM', 'BAC', 'GS', 'V', 'MA', 'BLK'],
    "ENERGY": ['XOM', 'CVX', 'SHELL', 'BP', 'OXY'],
    "WATCHLIST": ['NVDA', 'BTC-USD', 'TSLA', 'PLTR', 'COIN'] 
}

# --- THE INDUS VC ENGINE (CORE LOGIC) ---
class IndusVCEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.info = {}
        self.sentiment_score = 0
        self.crypto_fng = None 
        self.model = None
        self.features = [] 
        self.fundamentals = {} 
        self.win_rate = 0.0 

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period="5y")
        
        try:
            self.info = stock.info
            self.extract_fundamentals()
        except:
            pass

        if self.data.empty:
            raise ValueError(f"No data")

        self.data.reset_index(inplace=True)
        self.data['Date'] = self.data['Date'].dt.tz_localize(None)
        self.data.set_index('Date', inplace=True)
        
        # Macro Data
        try:
            macro_tickers = ['SPY', '^VIX']
            macro_data = yf.download(macro_tickers, period="5y", progress=False)['Close']
            self.data = self.data.join(macro_data)
            self.data.rename(columns={'SPY': 'SPY_Close', '^VIX': 'VIX_Close'}, inplace=True)
            self.data = self.data.ffill() 
            self.data.dropna(inplace=True)
        except:
            pass 

        if "-USD" in self.ticker:
            self.fetch_crypto_fng()

    def analyze_sentiment(self):
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news
            if not news: return
            analyzer = SentimentIntensityAnalyzer()
            scores = [analyzer.polarity_scores(a.get('title', ''))['compound'] for a in news]
            if scores: self.sentiment_score = sum(scores) / len(scores)
        except:
            pass

    def extract_fundamentals(self):
        if "-USD" in self.ticker: return
        try:
            # We now fetch MANY more metrics for the dashboard
            i = self.info
            self.fundamentals = {
                "PE": i.get('trailingPE', None),
                "Margins": i.get('profitMargins', None),
                "MarketCap": i.get('marketCap', None),
                "Sector": i.get('sector', 'Unknown'),
                "Industry": i.get('industry', 'Unknown'),
                "High52": i.get('fiftyTwoWeekHigh', None),
                "Low52": i.get('fiftyTwoWeekLow', None),
                "Dividend": i.get('dividendYield', None),
                "DebtToEquity": i.get('debtToEquity', None)
            }
        except:
            pass

    def fetch_crypto_fng(self):
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            data = requests.get(url).json()
            self.crypto_fng = (int(data['data'][0]['value']), data['data'][0]['value_classification'])
        except:
            pass

    def engineer_features(self):
        df = self.data.copy()
        df['Target'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        self.features = ['RSI', 'Dist_SMA_50', 'ATR'] 
        if 'VIX_Close' in df.columns:
            df['VIX_Level'] = df['VIX_Close']
            self.features.append('VIX_Level')

        for lag in [1, 2, 3]:
            df[f'Return_Lag_{lag}'] = df['Target'].shift(lag)
            self.features.append(f'Return_Lag_{lag}')
        
        df.dropna(inplace=True)
        self.data = df

    def train_model(self):
        split = int(len(self.data) * 0.8)
        self.train_data = self.data.iloc[:split]
        self.test_data = self.data.iloc[split:]
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=4, n_jobs=-1)
        self.model.fit(self.train_data[self.features], self.train_data['Target'])

    def get_forecast(self):
        latest = self.data.iloc[[-1]][self.features]
        pred_log = self.model.predict(latest)[0]
        return (np.exp(pred_log) - 1) * 100

    def calculate_kelly(self, win_rate, reward_risk_ratio=1.5):
        if win_rate <= 0.5: return 0.0 
        p = win_rate
        q = 1 - p
        b = reward_risk_ratio
        return max(0.0, ((b * p - q) / b) * 0.5 * 100)

    def run_backtest(self):
        initial_capital = 10000
        cash = initial_capital
        portfolio_values = []
        
        X_test = self.test_data[self.features]
        preds_pct = (np.exp(self.model.predict(X_test)) - 1) * 100
        prices = self.test_data['Close'].values
        dates = self.test_data.index
        
        wins = 0
        losses = 0
        
        for i in range(len(self.test_data) - 1):
            if preds_pct[i] > 0.5 and cash > 0:
                cash = (cash / prices[i]) * prices[i+1] # Simulate holding for 1 day
                if prices[i+1] > prices[i]: wins += 1
                else: losses += 1
            elif preds_pct[i] < -0.5 and cash > 0:
                 # In this simple model we just stay in cash, so value doesn't change
                 pass
            portfolio_values.append(cash)
                
        total_trades = wins + losses
        self.win_rate = wins / total_trades if total_trades > 0 else 0
        
        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dates[:-1], portfolio_values, label='AI Strategy', color='#00D26A')
        
        # Buy Hold comparison
        buy_hold = (prices[:-1] / prices[0]) * initial_capital
        ax.plot(dates[:-1], buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f"Profit Simulation ($10k Start)")
        ax.legend()
        ax.grid(True, alpha=0.1)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        
        # Color axes for dark mode
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values(): spine.set_color('#30333d')

        return fig, portfolio_values[-1], self.win_rate

    def generate_analyst_report(self, pred_pct, rsi, vix, atr, curr, kelly_pct, sma_50):
        report = ""
        
        # 1. AI SIGNAL
        if pred_pct > 0.5: report += f"**🤖 AI Signal:** BUY (+{pred_pct:.2f}%) 🟢\n"
        elif pred_pct < -0.5: report += f"**🤖 AI Signal:** SELL ({pred_pct:.2f}%) 🔴\n"
        else: report += f"**🤖 AI Signal:** HOLD (Flat) 🟡\n"
        
        # 2. TREND & MOMENTUM (New!)
        trend = "UPTREND 📈" if curr > sma_50 else "DOWNTREND 📉"
        report += f"**📊 Trend:** {trend} (Price vs SMA50)\n"
        
        if rsi > 70: report += "⚠️ **Momentum:** Overbought (RSI > 70). Risk of pullback.\n"
        elif rsi < 30: report += "✅ **Momentum:** Oversold (RSI < 30). Potential bounce.\n"
        else: report += "ℹ️ **Momentum:** Neutral. Follow the trend.\n"

        # 3. VOLATILITY & RISK
        vol_state = "HIGH ⚡" if atr > (curr * 0.03) else "STABLE 🌊"
        report += f"**📉 Volatility:** {vol_state} (ATR: ${atr:.2f})\n"
        
        if vix > 30: report += "🚨 **Market Risk:** Extreme Fear. Reduce position sizes.\n"
        
        # 4. TRADE PLAN
        stop = curr - (2*atr)
        target = curr + (3*atr)
        risk_reward = 3/2 # Since target is 3ATR and stop is 2ATR
        
        report += "\n---\n**🛡️ TRADE EXECUTION PLAN:**\n"
        report += f"- **Entry Zone:** ${curr:.2f}\n"
        report += f"- **Stop Loss:** ${stop:.2f}\n"
        report += f"- **Profit Target:** ${target:.2f} (Risk/Reward: 1:1.5)\n"
        report += f"- **Kelly Bet Size:** {kelly_pct:.1f}% of Capital"
        
        return report

# --- STREAMLIT UI LOGIC ---

st.title("🚀 IndusVC: Institutional Quant Dashboard")
st.markdown("---")

# SIDEBAR
with st.sidebar:
    st.header("📡 Command Center")
    mode = st.radio("Select Mode:", ["Market Radar", "Deep Dive Analysis"])
    st.markdown("---")
    
    if mode == "Deep Dive Analysis":
        ticker_input = st.text_input("Enter Ticker:", value="NVDA").upper()
        run_btn = st.button("Run Full Analysis")
        
    elif mode == "Market Radar":
        sector = st.selectbox("Select Sector:", list(SECTORS.keys()))
        scan_btn = st.button(f"Scan {sector} Sector")

# MAIN AREA: DEEP DIVE
if mode == "Deep Dive Analysis" and run_btn:
    with st.spinner(f"Analyzing {ticker_input} (Fundamentals, Macro, AI)..."):
        try:
            # Initialize Engine
            engine = IndusVCEngine(ticker_input)
            engine.fetch_data()
            engine.analyze_sentiment()
            engine.engineer_features()
            engine.train_model()
            
            # Get Data
            curr_price = engine.data.iloc[-1]['Close']
            forecast_pct = engine.get_forecast()
            target_price = curr_price * (1 + forecast_pct/100)
            
            # 1. HEADLINES (METRICS)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${curr_price:.2f}")
            col2.metric("AI Forecast (24h)", f"{forecast_pct:+.2f}%", delta_color="normal")
            col3.metric("Target Price", f"${target_price:.2f}")
            
            # Crypto F&G or Sentiment
            if engine.crypto_fng:
                val, label = engine.crypto_fng
                col4.metric("Crypto F&G", f"{val}", label)
            else:
                sent_label = "Neutral"
                if engine.sentiment_score > 0.05: sent_label = "Positive 🟢"
                elif engine.sentiment_score < -0.05: sent_label = "Negative 🔴"
                col4.metric("News Sentiment", sent_label)

            # 2. CHARTS & BACKTEST
            st.subheader("📊 Performance & Simulation")
            fig, final_balance, win_rate = engine.run_backtest()
            st.pyplot(fig)
            
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Simulated Profit (1Yr)", f"${final_balance - 10000:.2f}")
            b_col2.metric("AI Win Rate", f"{win_rate*100:.1f}%")
            
            # Kelly Calculation
            kelly = engine.calculate_kelly(win_rate)
            b_col3.metric("Kelly Bet Size", f"{kelly:.1f}%")

            # 3. ANALYST REPORT (NOW WITH MORE INFO)
            st.subheader("📝 Quant Analyst Report")
            latest = engine.data.iloc[-1]
            
            # Pass SMA_50 to the report generator now
            report = engine.generate_analyst_report(
                forecast_pct, 
                latest['RSI'], 
                latest.get('VIX_Level', 0), 
                latest['ATR'], 
                curr_price, 
                kelly,
                latest['SMA_50']
            )
            st.info(report)
            
            # 4. FUNDAMENTALS EXPANDER (ENRICHED)
            if engine.fundamentals:
                with st.expander("📊 Fundamental Health Card", expanded=True):
                    # Show Sector info
                    st.caption(f"Sector: {engine.fundamentals.get('Sector')} | Industry: {engine.fundamentals.get('Industry')}")
                    
                    # Row 1
                    f1, f2, f3, f4 = st.columns(4)
                    
                    mcap = engine.fundamentals.get('MarketCap', 0)
                    if mcap: mcap_fmt = f"${mcap/1e9:.1f}B"
                    else: mcap_fmt = "N/A"
                    f1.metric("Market Cap", mcap_fmt)
                    
                    pe = engine.fundamentals.get('PE')
                    f2.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
                    
                    marg = engine.fundamentals.get('Margins')
                    f3.metric("Profit Margin", f"{marg*100:.1f}%" if marg else "N/A")
                    
                    div = engine.fundamentals.get('Dividend')
                    f4.metric("Dividend Yield", f"{div*100:.2f}%" if div else "0%")

                    st.markdown("---")
                    
                    # Row 2
                    f5, f6, f7, f8 = st.columns(4)
                    
                    high = engine.fundamentals.get('High52')
                    f5.metric("52W High", f"${high:.2f}" if high else "N/A")
                    
                    low = engine.fundamentals.get('Low52')
                    f6.metric("52W Low", f"${low:.2f}" if low else "N/A")
                    
                    debt = engine.fundamentals.get('DebtToEquity')
                    f7.metric("Debt/Equity", f"{debt:.2f}" if debt else "N/A")
                    
                    f8.metric("Status", "Active 🟢")

        except Exception as e:
            st.error(f"Analysis Failed: {e}")

# MAIN AREA: RADAR
elif mode == "Market Radar" and scan_btn:
    st.subheader(f"📡 Scanning {sector} Sector...")
    tickers = SECTORS[sector]
    
    leaderboard = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(tickers):
        try:
            eng = IndusVCEngine(t)
            eng.fetch_data()
            eng.engineer_features()
            eng.train_model()
            forecast = eng.get_forecast()
            
            sig = "HOLD"
            if forecast > 0.5: sig = "BUY"
            elif forecast < -0.5: sig = "SELL"
            
            leaderboard.append({
                "Ticker": t,
                "Price": f"${eng.data.iloc[-1]['Close']:.2f}",
                "Forecast": forecast,
                "Signal": sig
            })
        except:
            pass
        progress_bar.progress((i + 1) / len(tickers))
        
    # Sort and Display
    df = pd.DataFrame(leaderboard).sort_values(by="Forecast", ascending=False)
    
    # Styled Dataframe
    st.dataframe(
        df.style.map(lambda x: 'color: green' if x == 'BUY' else ('color: red' if x == 'SELL' else 'color: gray'), subset=['Signal'])
        .format({"Forecast": "{:+.2f}%"}),
        use_container_width=True
    )
    
    if not df.empty:
        top_pick = df.iloc[0]['Ticker']
        st.success(f"💡 **Top Pick:** {top_pick} shows the strongest momentum.")