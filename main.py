import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
import sys, os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION: SECTOR LISTS ---
SECTORS = {
    "1": ("TECH & AI", ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMD', 'GOOG', 'AMZN', 'META']),
    "2": ("CRYPTO", ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD']),
    "3": ("FINANCE", ['JPM', 'BAC', 'GS', 'V', 'MA', 'BLK']),
    "4": ("ENERGY", ['XOM', 'CVX', 'SHELL', 'BP', 'OXY']),
    "5": ("WATCHLIST", ['NVDA', 'BTC-USD', 'TSLA', 'PLTR', 'COIN']) 
}

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
        if verbose: print(f"1. 📡 Fetching data for {self.ticker}...")
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
        if verbose: print("   - 🌎 Merging Macro Data...")
        try:
            macro_tickers = ['SPY', '^VIX']
            macro_data = yf.download(macro_tickers, period="5y", progress=False)['Close']
            self.data = self.data.join(macro_data)
            self.data.rename(columns={'SPY': 'SPY_Close', '^VIX': 'VIX_Close'}, inplace=True)
            self.data = self.data.ffill() 
            self.data.dropna(inplace=True)
        except:
            pass 

        # Crypto Check
        if "-USD" in self.ticker and verbose:
            self.fetch_crypto_fng()

    def analyze_sentiment(self):
        """Scans news headlines for sentiment"""
        if verbose: print("   - 📰 Analyzing News Sentiment...")
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news
            if not news: return
            
            analyzer = SentimentIntensityAnalyzer()
            scores = []
            for article in news:
                title = article.get('title', '')
                scores.append(analyzer.polarity_scores(title)['compound'])
            
            if scores:
                self.sentiment_score = sum(scores) / len(scores)
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
        if verbose: print("   - 🪙 Fetching Fear & Greed...")
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            data = requests.get(url).json()
            self.crypto_fng = (int(data['data'][0]['value']), data['data'][0]['value_classification'])
        except:
            pass

    def engineer_features(self):
        if verbose: print("2. 🧠 Engineering Quant Features (ATR & Sharpe)...")
        df = self.data.copy()
        
        # Target
        df['Target'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Technicals
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # ATR (Volatility)
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
        if verbose: print("3. 🤖 Training XGBoost...")
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
        kelly_fraction = (b * p - q) / b
        safe_kelly = kelly_fraction * 0.5 # Half Kelly for safety
        return max(0.0, safe_kelly * 100)

    def run_backtest(self):
        print("4. ⏳ Running Profit Simulator (1 Year)...")
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
            move = preds_pct[i]
            if move > 0.5:
                actual_return = (prices[i+1] - prices[i]) / prices[i]
                if actual_return > 0: wins += 1
                else: losses += 1
                
        total_trades = wins + losses
        self.win_rate = wins / total_trades if total_trades > 0 else 0
        
        print(f"\n📊 BACKTEST STATS: Win Rate: {self.win_rate*100:.1f}% ({wins}/{total_trades} trades)")
        
        returns = pd.Series(prices).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        print(f"📊 SHARPE RATIO: {sharpe:.2f} (Risk-Adjusted Return)")

    def explain_decision(self, pred_pct, rsi, vix, atr, current_price, kelly_pct):
        explanation = []
        
        # 1. AI Signal
        if pred_pct > 0.5: explanation.append(f"AI predicts **GROWTH** (+{pred_pct:.2f}%).")
        elif pred_pct < -0.5: explanation.append(f"AI predicts **DROP** ({pred_pct:.2f}%).")
        else: explanation.append("AI predicts **FLAT** market.")

        # 2. Sentiment Signal
        if self.sentiment_score > 0.05: explanation.append("News sentiment is **Positive** 🟢.")
        elif self.sentiment_score < -0.05: explanation.append("News sentiment is **Negative** 🔴.")

        # 3. Fundamental Warning
        if self.fundamentals:
            pe = self.fundamentals.get('PE')
            if pe and pe > 80: explanation.append("⚠️ **Valuation Warning:** Stock is very expensive (High P/E).")

        # 4. Risk Plan
        stop_loss = current_price - (2 * atr) 
        target_price = current_price + (3 * atr) 
        explanation.append(f"🛡️ **Risk Plan:** Stop Loss @ **${stop_loss:.2f}**, Target @ **${target_price:.2f}**.")
        
        # 5. Position Sizing
        if pred_pct > 0.5:
            if kelly_pct > 0: explanation.append(f"💰 **Bet Size:** Allocate **{kelly_pct:.1f}%** of capital.")
            else: explanation.append("💰 **Bet Size:** Win rate too low, avoid trade.")

        return " ".join(explanation)

    def predict_tomorrow(self):
        latest = self.data.iloc[[-1]][self.features]
        curr = self.data.iloc[-1]['Close']
        atr = latest['ATR'].values[0] 
        pred_pct = self.get_forecast()
        kelly = self.calculate_kelly(self.win_rate)
        
        print("\n" + "="*60)
        print(f"  🔮 INDUS VC: {self.ticker} INTELLIGENCE REPORT")
        print("="*60)
        print(f"Current Price:   ${curr:.2f}")
        print(f"AI Forecast:     {pred_pct:+.2f}%")
        
        # --- QUANT METRICS ---
        print("-" * 60)
        print(f"📐 VOLATILITY (ATR): ${atr:.2f} (Daily Swing)")
        print(f"🛡️ STOP LOSS:       ${curr - (2*atr):.2f}")
        print(f"🎯 PROFIT TARGET:   ${curr + (3*atr):.2f}")
        print("-" * 60)
        
        if pred_pct > 0.5: print("🎯 SIGNAL: BUY 🟢")
        elif pred_pct < -0.5: print("🎯 SIGNAL: SELL 🔴")
        else: print("🎯 SIGNAL: HOLD 🟡")
        
        # --- CRYPTO SECTION ---
        if self.crypto_fng:
            val, label = self.crypto_fng
            print(f"🧠 CRYPTO MOOD: {val} ({label.upper()})")
            
        # --- FUNDAMENTALS ---
        if self.fundamentals and not "-USD" in self.ticker:
            pe = self.fundamentals.get('PE', 'N/A')
            marg = self.fundamentals.get('Margins', 0)
            if isinstance(pe, (int, float)): pe = f"{pe:.1f}"
            print(f"📊 HEALTH: P/E Ratio: {pe} | Margins: {marg*100:.1f}%")

        rsi = latest['RSI'].values[0]
        vix = latest['VIX_Level'].values[0] if 'VIX_Level' in latest.columns else 0
        
        print("-" * 60)
        print("📝 **ANALYST SUMMARY:**")
        print(self.explain_decision(pred_pct, rsi, vix, atr, curr, kelly))
        print("="*60)

# --- GLOBAL VARIABLES ---
verbose = True 

def run_radar(tickers):
    global verbose
    verbose = False 
    print(f"\n📡 SCANNING {len(tickers)} ASSETS...")
    leaderboard = []
    
    for t in tickers:
        try:
            print(f".", end="", flush=True) 
            eng = IndusVCEngine(t)
            eng.fetch_data()
            eng.engineer_features()
            eng.train_model()
            forecast = eng.get_forecast()
            
            sig = "HOLD 🟡"
            if forecast > 0.5: sig = "BUY 🟢"
            elif forecast < -0.5: sig = "SELL 🔴"
            
            leaderboard.append({'Ticker': t, 'Forecast': forecast, 'Signal': sig, 'Price': eng.data.iloc[-1]['Close']})
        except:
            pass
            
    leaderboard.sort(key=lambda x: x['Forecast'], reverse=True)
    
    print("\n\n" + "="*50)
    print(f"  🏆 TOP OPPORTUNITIES")
    print("="*50)
    print(f"{'TICKER':<10} {'PRICE':<10} {'FORECAST':<10} {'SIGNAL'}")
    print("-" * 50)
    for row in leaderboard[:5]:
        print(f"{row['Ticker']:<10} ${row['Price']:<9.2f} {row['Forecast']:>+5.2f}%    {row['Signal']}")
    return leaderboard

def run_direct_search():
    global verbose
    verbose = True
    ticker = input("\n🔍 Enter Stock Ticker: ").upper().strip()
    if ticker:
        try:
            engine = IndusVCEngine(ticker)
            engine.fetch_data()
            engine.analyze_sentiment() # Added Sentiment back!
            engine.engineer_features()
            engine.train_model()
            engine.run_backtest()
            engine.predict_tomorrow()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    while True:
        print("\n=== INDUS VC COMMAND CENTER ===")
        print("[1] Radar Scan  [2] Direct Search  [Q] Quit")
        mode = input("Select: ").upper().strip()
        if mode == 'Q': break
        elif mode == '2': run_direct_search()
        elif mode == '1':
            print("Select Sector Code (1-5):")
            for k,v in SECTORS.items(): print(f"[{k}] {v[0]}")
            c = input("Choice: ")
            if c in SECTORS: 
                ranked_list = run_radar(SECTORS[c][1])
                if ranked_list:
                    print(f"\n💡 Deep Dive on {ranked_list[0]['Ticker']}? (Y/N)")
                    if input().upper() == 'Y':
                        # Auto-run deep dive on top pick
                        verbose = True
                        eng = IndusVCEngine(ranked_list[0]['Ticker'])
                        eng.fetch_data()
                        eng.analyze_sentiment()
                        eng.engineer_features()
                        eng.train_model()
                        eng.run_backtest()
                        eng.predict_tomorrow()


                        