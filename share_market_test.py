import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import ta
import json
from pathlib import Path

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDfXA0kfutmf6T4ftsxOQPg0sLIXcTBA3E"
genai.configure(api_key=GEMINI_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

class RiskManager:
    def __init__(self):
        self.var_confidence = 0.99  # 99% VaR
        self.max_portfolio_risk = 0.15  # 15% max risk
        
    def calculate_var(self, returns, position_size):
        """Calculate Value at Risk"""
        var = stats.norm.ppf(1 - self.var_confidence) * np.std(returns) * position_size
        return abs(var)
        
    def calculate_expected_shortfall(self, returns, position_size):
        """Calculate Expected Shortfall (CVaR)"""
        var = self.calculate_var(returns, position_size)
        return np.mean(returns[returns <= -var]) * position_size
        
    def get_position_sizing(self, capital, volatility, correlation_matrix):
        """Kelly Criterion with correlation adjustment"""
        kelly_fraction = (1 / volatility) * (1 - np.mean(correlation_matrix))
        return min(kelly_fraction * capital, capital * 0.25)  # Max 25% per position

class TradingAssistant:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.market_data = None
        self.last_analysis = None
        self.history_file = Path("chat_history.json")
        self.load_chat_history()
        self.feedback_file = Path("feedback_data_share.json")
        self.feedback_data = self.load_feedback_data()
        self.risk_manager = RiskManager()  # Initialize risk manager

    def load_chat_history(self):
        """Load chat history from JSON file"""
        try:
            if self.history_file.exists():
                try:
                    with open(self.history_file, 'r') as f:
                        history = json.load(f)
                        messages = history.get('messages', [])
                        
                        # Reconstruct Plotly figures from saved data
                        for message in messages:
                            if "chart_data" in message:
                                try:
                                    chart_config = message["chart_data"]
                                    fig = go.Figure()
                                    
                                    # Add candlestick chart
                                    fig.add_trace(go.Candlestick(
                                        x=chart_config["index"],
                                        open=chart_config["open"],
                                        high=chart_config["high"],
                                        low=chart_config["low"],
                                        close=chart_config["close"],
                                        name=message.get("role", "Market Data")
                                    ))
                                    
                                    # Add moving averages
                                    fig.add_trace(go.Scatter(
                                        x=chart_config["index"],
                                        y=chart_config["ma50"],
                                        name="50-day MA",
                                        line=dict(color='orange')
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=chart_config["index"],
                                        y=chart_config["ma200"],
                                        name="200-day MA",
                                        line=dict(color='blue')
                                    ))
                                    
                                    fig.update_layout(
                                        title="Technical Chart",
                                        yaxis_title="Price",
                                        xaxis_title="Date",
                                        height=600
                                    )
                                    
                                    message["chart_data"] = fig
                                except Exception as chart_error:
                                    st.warning(f"Error reconstructing chart: {chart_error}")
                                    message.pop("chart_data", None)
                        
                        st.session_state.messages = messages
                except json.JSONDecodeError as e:
                    st.error(f"Invalid chat history file format. Creating new history. Error: {e}")
                    # Backup corrupted file
                    if self.history_file.exists():
                        backup_file = self.history_file.with_suffix('.bak')
                        self.history_file.rename(backup_file)
                    st.session_state.messages = []
                    # Create new history file
                    self.save_chat_history()
            else:
                st.session_state.messages = []
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            st.session_state.messages = []

    def save_chat_history(self):
        """Save chat history to JSON file"""
        try:
            # Create a copy of messages without Plotly figures
            serializable_messages = []
            for message in st.session_state.messages:
                msg_copy = message.copy()
                if "chart_data" in msg_copy:
                    # Convert datetime index to string format
                    chart_data = message["chart_data"].data[0]
                    # Store only the chart configuration instead of the Figure object
                    msg_copy["chart_data"] = {
                    "type": "candlestick",
                    "index": [d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d) 
                             for d in chart_data.x],
                    "open": [float(x) for x in chart_data.open],
                    "high": [float(x) for x in chart_data.high],
                    "low": [float(x) for x in chart_data.low],
                    "close": [float(x) for x in chart_data.close],
                    "ma50": [float(x) for x in message["chart_data"].data[1].y],
                    "ma200": [float(x) for x in message["chart_data"].data[2].y]
                }
                serializable_messages.append(msg_copy)

            history = {
                'messages': serializable_messages,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")   

    def load_feedback_data(self):
        """Load feedback history from JSON file"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    # Ensure all required keys exist
                    if not all(key in data for key in ['positive', 'negative', 'improvements']):
                        data = {
                            'positive': data.get('positive', []),
                            'negative': data.get('negative', []),
                            'improvements': data.get('improvements', [])
                        }
                    return data
            # Return default structure if file doesn't exist
            return {
                'positive': [],
                'negative': [],
                'improvements': []
            }
        except Exception as e:
            st.error(f"Error loading feedback data: {e}")
            # Return default structure on error
            return {
                'positive': [],
                'negative': [],
                'improvements': []
            }

    def save_feedback(self, query, response, feedback, improvement=None):
        """Save user feedback and use it to improve responses"""
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'improvement': improvement
            }
            
            if feedback == 'positive':
                self.feedback_data['positive'].append(feedback_entry)
            else:
                self.feedback_data['negative'].append(feedback_entry)
                if improvement:
                    self.feedback_data['improvements'].append(improvement)
            
            # Save feedback data
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving feedback: {e}") 
        
    def load_market_data(self):
        """Load market data from Excel file"""
        try:
            self.market_data = pd.read_excel("market_data_5y.xlsx", index_col=0)
            self.market_data.index = pd.to_datetime(self.market_data.index)
            return True
        except Exception as e:
            st.error(f"Error loading market data: {e}")
            return False
        
    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        hl2 = (df['High'] + df['Low']) / 2
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
        
        # Calculate SuperTrend bands
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        # First value
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        # Calculate SuperTrend values
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
                
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = min(lower_band.iloc[i], supertrend.iloc[i-1])
            else:
                supertrend.iloc[i] = max(upper_band.iloc[i], supertrend.iloc[i-1])
        
        return direction
    
    def calculate_metrics(self, df, index_name):
        """Calculate institutional-grade metrics"""
        metrics = {}
        df_index = df[df['Index'] == index_name].copy()
        
        # Advanced technical indicators
        df_index['EMA21'] = ta.trend.EMAIndicator(df_index['Close'], 21).ema_indicator()
        df_index['SuperTrend'] = self.calculate_supertrend(df_index)
        
        # Liquidity metrics
        df_index['ADR'] = (df_index['High'] - df_index['Low']) / df_index['Low'] * 100
        df_index['Volume_MA20'] = df_index['Volume'].rolling(window=20).mean()
        metrics['liquidity_score'] = (
            df_index['Volume'].iloc[-1] / df_index['Volume_MA20'].iloc[-1]
        )
        
        # Institutional money flow
        df_index['MFI'] = ta.volume.MFIIndicator(
            df_index['High'], df_index['Low'], 
            df_index['Close'], df_index['Volume']
        ).money_flow_index()
        
        # Basic metrics
        metrics['current_price'] = df_index['Close'].iloc[-1]
        metrics['prev_close'] = df_index['Close'].iloc[-2]
        metrics['daily_return'] = ((metrics['current_price'] - metrics['prev_close'])/metrics['prev_close']) * 100
        
        # Technical indicators
        df_index['MA20'] = df_index['Close'].rolling(window=20).mean()
        df_index['MA50'] = df_index['Close'].rolling(window=50).mean()
        df_index['MA200'] = df_index['Close'].rolling(window=200).mean()
        
        # Volatility
        df_index['Daily_Return'] = df_index['Close'].pct_change()
        metrics['volatility'] = df_index['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Support and Resistance
        metrics['support'] = df_index['Low'].tail(20).min()
        metrics['resistance'] = df_index['High'].tail(20).max()
        
        # Trend strength
        metrics['trend'] = "Bullish" if metrics['current_price'] > df_index['MA200'].iloc[-1] else "Bearish"
        
        return metrics, df_index
    
    def update_features(self, current_features, predicted_price):
        """Update feature vector for next prediction"""
        try:
            # Convert to numpy array if not already
            features = np.array(current_features)
            
            # Update Close price (index 3 in our feature list)
            features[0, 3] = predicted_price
            
            # Update moving averages (indices 6, 7, 8 for MA20, MA50, MA200)
            features[0, 6] = (features[0, 6] * 19 + predicted_price) / 20  # MA20
            features[0, 7] = (features[0, 7] * 49 + predicted_price) / 50  # MA50
            features[0, 8] = (features[0, 8] * 199 + predicted_price) / 200  # MA200
            
            # Update RSI (simplified calculation)
            current_rsi = features[0, 9]
            features[0, 9] = max(0, min(100, current_rsi + np.random.normal(0, 2)))
            
            # Update other technical indicators with reasonable approximations
            features[0, 10:] = features[0, 10:]  # Keep other indicators unchanged for simplicity
            
            return features
            
        except Exception as e:
            st.error(f"Error updating features: {e}")
            return current_features
    
    def predict_future_price(self, df_index, days=7):
        """Predict future prices using ML and technical analysis"""
        try:
            # Prepare features for prediction
            df = df_index.copy()
            
            # Add technical indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd_diff()
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            # Create features
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                             'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                             'BB_upper', 'BB_middle', 'BB_lower', 'ADX']
            
            # Handle NaN values
            df = df.fillna(method='ffill')
            
            # Scale features
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(df[feature_columns])
            
            # Prepare training data
            X = features_scaled[:-1]  # All data except last row
            y = df['Close'].values[1:]  # Target is next day's close price
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Predict next 7 days
            last_features = features_scaled[-1:]
            predictions = []
            current_features = last_features.copy()
            
            for _ in range(days):
                next_price = model.predict(current_features)[0]
                predictions.append(next_price)
                
                # Update features for next prediction
                current_features = self.update_features(current_features, next_price)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None
    
    def generate_analysis(self, query, index_name="SENSEX"):
        """Generate comprehensive analysis based on user query with context"""
        if self.market_data is None:
            return "No market data available. Please load data first."
            
        metrics, df_index = self.calculate_metrics(self.market_data, index_name)
        
        # Calculate risk metrics
        risk_metrics = {
            'var': self.risk_manager.calculate_var(
                df_index['Daily_Return'].dropna(), 
                metrics['current_price']
            ),
            'cvar': self.risk_manager.calculate_expected_shortfall(
                df_index['Daily_Return'].dropna(),
                metrics['current_price']
            ),
            'position_size': self.risk_manager.get_position_sizing(
                capital=300000000,  # 30 crores
                volatility=metrics['volatility'] / 100,
                correlation_matrix=df_index['Close'].corr(df_index['Volume'])
            )
        }
        
        # Add risk manager instance if not exists
        if not hasattr(self, 'risk_manager'):
            self.risk_manager = RiskManager()

        # Get previous analyses for context
        previous_analyses = self._get_relevant_history(query)
        
        # Add feedback context to prompt
        feedback_context = self._get_feedback_context()

        # Add previous context to prompt
        context = self._build_context(previous_analyses)

        # Get future predictions if query is about future prices
        future_context = ""
        if any(word in query.lower() for word in ['next week', 'future', 'predict', 'prediction', 'forecast']):
            predictions = self.predict_future_price(df_index)
            if predictions:
                current_price = metrics['current_price']
                predicted_prices = [f"Day {i+1}: ₹{price:,.2f} ({((price-current_price)/current_price)*100:+.2f}%)"
                                  for i, price in enumerate(predictions)]
                future_context = f"""
                Predicted prices for next week:
                {chr(10).join(predicted_prices)}
                """

        prompt = f"""
        Act as an institutional-grade quantitative strategist managing a ₹30+ crore portfolio. 
        You have access to comprehensive market data and risk metrics for {index_name}.
        2. Previous analysis context:
        {context}
        3. User feedback insights:
        {feedback_context}

        Current date: {datetime.now().strftime('%Y-%m-%d')}

        Current Market Metrics:
        - Current Price: ₹{metrics['current_price']:,.2f}
        - Daily Change: {metrics['daily_return']:.2f}%
        - Market Trend: {metrics['trend']}
        - Volatility (Annual): {metrics['volatility']:.2f}%
        - Support Level: ₹{metrics['support']:,.2f}
        - Resistance Level: ₹{metrics['resistance']:,.2f}
        - Liquidity Score: {metrics['liquidity_score']:.2f}
        - Money Flow Index: {df_index['MFI'].iloc[-1]:.2f}
        - SuperTrend Signal: {df_index['SuperTrend'].iloc[-1]}
        - Risk Metrics:
        * Value at Risk (99%): ₹{risk_metrics['var']:,.2f}
        * Expected Shortfall: ₹{risk_metrics['cvar']:,.2f}
        * Position Size Recommendation: ₹{risk_metrics['position_size']:,.2f}

        Portfolio Context:
        - AUM: several crores
        - Risk Tolerance: Moderate-Aggressive
        - Investment Horizon: Strategic (1-3 years) and Tactical (1-6 months)
        - Compliance Requirements: SEBI regulations for portfolio management

        {future_context}
        
        User Query: {query}
        
        Consider previous analyses and provide updated insights.
        Include any changes in outlook from previous predictions.

        Based on the above metrics, provide:
        1. Strategic Portfolio Positioning
        - Asset Allocation
        - Sector Rotation Strategy
        - Risk-Adjusted Position Sizing
        
        2. Technical Analysis
        - Key Price Levels
        - Volume Profile Analysis
        - Institutional Money Flow
        
        3. Risk Management
        - Position-Level Stop Loss
        - Portfolio-Level Hedging
        - Correlation with Existing Holdings
        
        4. Regulatory Compliance
        - SEBI Exposure Norms
        - Concentration Risk
        - Disclosure Requirements
        
        Format as a professional investment memo with clear action items.
        Include specific entry/exit levels and position sizing recommendations.
        Consider regulatory compliance and risk management for large positions.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating analysis: {e}"
        
    def _get_relevant_history(self, query, max_items=3):
        """Get relevant previous analyses"""
        relevant = []
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant":
                relevant.append(message["content"])
                if len(relevant) >= max_items:
                    break
        return relevant
    
    def _build_context(self, previous_analyses):
        """Build context string from previous analyses"""
        if not previous_analyses:
            return "No previous analyses available."
        
        context = "Previous analyses:\n"
        for i, analysis in enumerate(reversed(previous_analyses), 1):
            context += f"\nAnalysis {i}:\n{analysis[:500]}...\n"  # Truncate long analyses
        return context
    
    def _get_feedback_context(self):
        """Generate context from past feedback"""
        try:
            # Ensure improvements key exists
            improvements = self.feedback_data.get('improvements', [])
            
            if not improvements:
                return "No feedback data available."
            
            # Get last 3 improvements
            recent_improvements = improvements[-3:]
            context = "Recent feedback suggestions:\n"
            for imp in recent_improvements:
                if isinstance(imp, str):
                    context += f"- {imp}\n"
                elif isinstance(imp, dict) and 'improvement' in imp:
                    context += f"- {imp['improvement']}\n"
            return context
        except Exception as e:
            st.error(f"Error getting feedback context: {e}")
            return "Error retrieving feedback data."
        
class PortfolioManager:
    def __init__(self, aum=300000000):  # 30 crores
        self.aum = aum
        self.positions = {}
        self.risk_manager = RiskManager()
        self.max_position_size = 0.25  # 25% of AUM
        self.sector_limits = 0.35  # 35% per sector
        
    def add_position(self, symbol, amount, sector):
        """Add new position with compliance checks"""
        if self._check_compliance(symbol, amount, sector):
            self.positions[symbol] = {
                'amount': amount,
                'sector': sector,
                'entry_price': self._get_current_price(symbol),
                'timestamp': datetime.now()
            }
            return True
        return False
        
    def _check_compliance(self, symbol, amount, sector):
        """Check SEBI compliance for position"""
        # Position size check
        if amount > self.aum * self.max_position_size:
            return False
            
        # Sector exposure check
        sector_exposure = sum(
            pos['amount'] for pos in self.positions.values() 
            if pos['sector'] == sector
        )
        if (sector_exposure + amount) > self.aum * self.sector_limits:
            return False
            
        return True

def main():
    st.set_page_config(page_title="Smart Trading Assistant", layout="wide")
    
    st.title("🤖 Smart Trading Assistant")
    
    # Initialize trading assistant
    assistant = TradingAssistant()
    
    # Load market data
    if assistant.load_market_data():
        st.success("Market data loaded successfully!")
        
        # Sidebar controls
        st.sidebar.header("Analysis Settings")
        index_choice = st.sidebar.radio("Select Index", ["SENSEX", "NIFTY 50", "Reliance", "TCS", "HDFC Bank", "SBI", 
                                                         "Rail Vikas Nigam", "Cochin Shipyard", "Exide Industries",
                                                         "Tata Motors", "ITC", "Hindustan Unilever", "HUDCO",
                                                         "GE Vernova T&D", "Power Finance Corporation", "EIH Limited",
                                                         "Carborundum Universal", "Bharat Heavy Electricals Limited",
                                                         "Bharat Electronics", "Bharat Dynamics", "Bajaj Housing Finance"])
        
        # Clear chat button
        if st.sidebar.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "chart_data" in message:
                    # Add unique key for chat history charts
                    st.plotly_chart(message["chart_data"], 
                                use_container_width=True,
                                key=f"chat_chart_{idx}")

        # Chat input
        if prompt := st.chat_input("Ask me about the market..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # Get analysis
                    analysis = assistant.generate_analysis(prompt, index_choice)

                    # Display analysis
                    st.markdown(analysis)
                    
                    # Create and display technical chart
                    metrics, df_index = assistant.calculate_metrics(assistant.market_data, index_choice)
                    
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=df_index.index,
                        open=df_index['Open'],
                        high=df_index['High'],
                        low=df_index['Low'],
                        close=df_index['Close'],
                        name=index_choice
                    ))
                    
                    # Add moving averages
                    fig.add_trace(go.Scatter(x=df_index.index, y=df_index['MA50'], 
                                           name="50-day MA", line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=df_index.index, y=df_index['MA200'], 
                                           name="200-day MA", line=dict(color='blue')))
                    
                    fig.update_layout(
                        title=f"{index_choice} Technical Chart",
                        yaxis_title="Price",
                        xaxis_title="Date",
                        height=600
                    )
                    
                    # Add unique key for current analysis chart
                    st.plotly_chart(fig, 
                              use_container_width=True,
                              key=f"analysis_chart_{datetime.now().timestamp()}")

                    # Add feedback section
                    with st.container():
                        st.markdown("---")
                        st.markdown("### Was this analysis helpful?")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("👍 Yes", key=f"positive_{datetime.now().timestamp()}"):
                                assistant.save_feedback(prompt, analysis, 'positive')
                                st.success("Thank you for your feedback!")
                        
                        with col2:
                            if st.button("👎 No", key=f"negative_{datetime.now().timestamp()}"):
                                improvement = st.text_area(
                                    "What could be improved?",
                                    key=f"improvement_{datetime.now().timestamp()}"
                                )
                                if st.button("Submit Feedback", key=f"submit_{datetime.now().timestamp()}"):
                                    assistant.save_feedback(prompt, analysis, 'negative', improvement)
                                    st.success("Thank you for your feedback! We'll use it to improve.")

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": analysis,
                        "chart_data": fig
                    })

                    # Save updated chat history
                    assistant.save_chat_history()

if __name__ == "__main__":
    main()