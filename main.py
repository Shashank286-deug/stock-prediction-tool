import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

class IndusVCEngine:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.info = {}
        self.sentiment_score = 0
        self.model = None
        self.features = [] 

    def fetch_data(self):
        """Fetches Stock Data, Company Profile, AND Macro Context"""
        print(f"1. Fetching data for {self.ticker}...")
        
        # A. Main Stock Data (5 Years)
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period="5y")
        
        try:
            self.info = stock.info
        except:
            print("   - Warning: Could not fetch company profile.")

        if self.data.empty:
            raise ValueError(f"ERROR: No data found for '{self.ticker}'.")

        # Clean Main Data
        self.data.reset_index(inplace=True)
        self.data['Date'] = self.data['Date'].dt.tz_localize(None)
        self.data.set_index('Date', inplace=True)
        
        # B. Macro Context
        print("   - Merging Macro Economic Data (SPY, VIX, Gold)...")
        macro_tickers = ['SPY', '^VIX', 'GC=F']
        
        try:
            # Fetch macro data
            macro_data = yf.download(macro_tickers, period="5y", progress=False)['Close']
            
            # Merge Macro data into main dataframe
            self.data = self.data.join(macro_data)
            
            # Rename columns
            self.data.rename(columns={
                'SPY': 'SPY_Close', 
                '^VIX': 'VIX_Close',
                'GC=F': 'Gold_Close'
            }, inplace=True)
            
            # --- FIX FOR PANDAS 2.0+ ---
            # Replaced deprecated fillna(method='ffill') with ffill()
            self.data = self.data.ffill()
            
            self.data.dropna(inplace=True)
            print(f"   - Data Ready: {len(self.data)} rows.")
            
        except Exception as e:
            print(f"   - Warning: Macro data merge failed ({e}). Proceeding with stock data only.")
            # If macro fails, we just continue with the stock data to prevent crashing
            self.data = self.data.dropna()

    def analyze_sentiment(self):
        print("2. Scanning News Sentiment...")
        stock = yf.Ticker(self.ticker)
        news_list = stock.news
        
        if not news_list:
            self.sentiment_score = 0
            return

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        
        for article in news_list:
            title = article.get('title', '')
            score = analyzer.polarity_scores(title)['compound']
            scores.append(score)
            
        if scores:
            self.sentiment_score = sum(scores) / len(scores)
            
        if self.sentiment_score > 0.05: mood = "Positive 🟢"
        elif self.sentiment_score < -0.05: mood = "Negative 🔴"
        else: mood = "Neutral 🟡"
        print(f"   - Market Mood: {mood} (Score: {self.sentiment_score:.2f})")

    def engineer_features(self):
        print("3. Engineering Features (Math & Memory)...")
        df = self.data.copy()
        
        # Target: Log Returns
        df['Target'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Technicals
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        
        # Avoid division by zero
        df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Initialize feature list
        self.features = ['RSI', 'Dist_SMA_50']
        
        # Add Macro features if they exist
        if 'VIX_Close' in df.columns:
            df['VIX_Level'] = df['VIX_Close']
            df['SPY_Return'] = df['SPY_Close'].pct_change()
            self.features.extend(['VIX_Level'])
            
            # Lag Features for SPY
            for lag in [1, 2, 3]:
                df[f'SPY_Return_Lag_{lag}'] = df['SPY_Return'].shift(lag)
                self.features.append(f'SPY_Return_Lag_{lag}')

        # Lag Features for Main Stock
        for lag in [1, 2, 3]:
            df[f'Return_Lag_{lag}'] = df['Target'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
            self.features.append(f'Return_Lag_{lag}')
        
        df.dropna(inplace=True)
        self.data = df

    def train_xgboost(self):
        print("4. Training XGBoost AI...")
        if len(self.data) < 100:
            raise ValueError("Not enough data to train. Try a different stock.")
            
        split = len(self.data) - 50 # Test on last 50 days
        train = self.data.iloc[:split]
        test = self.data.iloc[split:]
        
        X_train, y_train = train[self.features], train['Target']
        X_test, y_test = test[self.features], test['Target']
        
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        correct_direction = np.sign(preds) == np.sign(y_test)
        acc = np.mean(correct_direction) * 100
        print(f"   - Directional Accuracy (Test Set): {acc:.1f}%")

    def generate_chart(self):
        print("5. Generating Chart...")
        recent = self.data.tail(180)
        plt.figure(figsize=(10, 5))
        plt.plot(recent.index, recent['Close'], label='Price', color='black')
        plt.plot(recent.index, recent['SMA_50'], label='Trend (SMA 50)', color='green', linestyle='--')
        plt.title(f"{self.ticker} Analysis (Last 6 Months)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("stock_analysis.png")
        plt.close()
        print("   - Chart saved as 'stock_analysis.png'")

    def predict(self):
        latest = self.data.iloc[[-1]][self.features]
        curr_price = self.data.iloc[-1]['Close']
        pred_log_return = self.model.predict(latest)[0]
        pred_price = curr_price * np.exp(pred_log_return)
        pct_change = (np.exp(pred_log_return) - 1) * 100
        return curr_price, pred_price, pct_change

if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (e.g., TSLA, NVDA): ").upper().strip()
    
    try:
        # Initialize Engine
        engine = IndusVCEngine(ticker)
        
        # Run Pipeline
        engine.fetch_data()
        engine.analyze_sentiment()
        engine.engineer_features()
        engine.train_xgboost()
        engine.generate_chart()
        
        # Get Results
        curr, pred, chg = engine.predict()
        
        # --- THE FINAL REPORT ---
        print("\n" + "="*60)
        print(f"  INDUS VC ANALYTICS: {ticker}")
        print("="*60)
        print(f"Sector: {engine.info.get('sector', 'N/A')} | Market Cap: ${engine.info.get('marketCap', 0):,}")
        print("-" * 60)
        print(f"Current Price:   ${curr:.2f}")
        print(f"Predicted Price: ${pred:.2f} ({chg:.2f}%)")
        print("-" * 60)
        
        # DECISION LOGIC
        signal = "HOLD 🟡"
        if chg > 0.5: tech = "BUY"
        elif chg < -0.5: tech = "SELL"
        else: tech = "NEUTRAL"
        
        sent_score = engine.sentiment_score
        if sent_score > 0.05: sent = "POSITIVE"
        elif sent_score < -0.05: sent = "NEGATIVE"
        else: sent = "NEUTRAL"
        
        # Check if VIX exists (macro data might have failed)
        if 'VIX_Level' in engine.data.columns:
            vix = engine.data.iloc[-1]['VIX_Level']
        else:
            vix = 0
        
        print(f"• AI Model:       {tech}")
        print(f"• News Sentiment: {sent}")
        if vix > 0:
            print(f"• Market Risk (VIX): {vix:.2f}")
        print("-" * 60)
        
        if tech == "BUY" and sent == "POSITIVE" and (vix < 30 or vix == 0):
            print("FINAL VERDICT: STRONG BUY 🟢🟢")
        elif tech == "SELL" and sent == "NEGATIVE":
            print("FINAL VERDICT: STRONG SELL 🔴🔴")
        elif vix > 30:
            print("FINAL VERDICT: DO NOT TRADE (High Risk) ⛔")
        else:
            print(f"FINAL VERDICT: {tech}")
        print("="*60)
        print("Note: Check 'stock_analysis.png' for the chart.")
        
    except Exception as e:
        print(f"\nError: {e}")

        