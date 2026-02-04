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
            self.fundamentals = {
                "PE": self.info.get('trailingPE', None),
                "Margins": self.info.get('profitMargins', None),
                "MarketCap": self.info.get('marketCap', None)
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

    def generate_analyst_report(self, pred_pct, rsi, vix, atr, curr, kelly_pct):
        report = ""
        
        # Signal
        if pred_pct > 0.5: report += f"**AI Signal:** BUY (+{pred_pct:.2f}%) 🟢\n\n"
        elif pred_pct < -0.5: report += f"**AI Signal:** SELL ({pred_pct:.2f}%) 🔴\n\n"
        else: report += f"**AI Signal:** HOLD (Flat) 🟡\n\n"
        
        # Context
        if rsi > 70: report += "⚠️ **Momentum:** Asset is Overbought (RSI > 70). Risk of pullback.\n"
        elif rsi < 30: report += "✅ **Momentum:** Asset is Oversold (RSI < 30). Potential bounce opportunity.\n"
        
        if vix > 30: report += "🚨 **Market Risk:** Extreme Fear in the market. Reduce position sizes.\n"
        
        # Fundamentals
        if self.fundamentals:
