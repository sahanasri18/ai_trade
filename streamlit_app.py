import streamlit as st
import yfinance as yf
import pandas as pd
from environment import TradingEnvironment
from data_utils import preprocess_data

st.set_page_config(page_title="AI Stock Trader", page_icon="📈")

st.markdown("<h1 style='text-align:center;'>📈 AI-Powered Stock Trading Dashboard</h1>", unsafe_allow_html=True)

# 1. Predefined dropdown symbols
popular_symbols = {
    "📉 Bitcoin (BTC)": "BTC-USD",
    "📈 Apple (AAPL)": "AAPL",
    "🏦 Infosys (INFY India)": "INFY.BO",
    "💼 Microsoft (MSFT)": "MSFT",
    "📊 Nifty 50 (India)": "^NSEI",
}

symbol_name = st.selectbox("Select Asset:", list(popular_symbols.keys()))
symbol = popular_symbols[symbol_name]

# 2. Load and preprocess data
@st.cache_data
def load_data(sym):
    df = yf.download(sym, start="2020-01-01", end="2023-01-01")
    return preprocess_data(df)

df = load_data(symbol)

if df.empty:
    st.error("❌ No data found.")
    st.stop()

env = TradingEnvironment(df)

# 3. Session state to persist across interactions
if "state" not in st.session_state:
    st.session_state.state = env.reset()
    st.session_state.balance = env.initial_balance
    st.session_state.shares = 0
    st.session_state.history = []
    st.session_state.step = 0
    st.session_state.net_worth = env.initial_balance

# 4. Action buttons
col1, col2, col3 = st.columns(3)
buy = col1.button("🟢 Buy")
sell = col2.button("🔴 Sell")
hold = col3.button("🟡 Hold")

action_map = {buy: 1, sell: 2, hold: 0}
action_taken = None
for key, val in action_map.items():
    if key:
        action_taken = val
        break

# 5. Step the environment
if action_taken is not None:
    prev_price = df.iloc[st.session_state.step]['Close']
    prev_balance = st.session_state.balance
    prev_shares = st.session_state.shares

    state, reward, done, _ = env.step(action_taken)

    st.session_state.state = state
    st.session_state.step = env.current_step
    st.session_state.balance = env.balance
    st.session_state.shares = env.shares_held
    st.session_state.net_worth = env.net_worth

    price = float(df.iloc[env.current_step]["Close"])

    action_label = ["Hold", "Buy", "Sell"][action_taken]

    if action_taken == 1 and prev_balance > price:
        st.success(f"🟢 Bought 1 share at ${price:.2f}")
    elif action_taken == 2 and prev_shares > 0:
        st.warning(f"🔴 Sold 1 share at ${price:.2f}")
    elif action_taken == 0:
        st.info(f"🟡 Holding...")

    # Save history
    st.session_state.history.append({
        "Step": st.session_state.step,
        "Action": action_label,
        "Price": f"${price:.2f}",
        "Net Worth": f"${st.session_state.net_worth:.2f}"
    })

# 6. Show trade history table
if st.session_state.history:
    st.markdown("### 📋 Trade History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# 7. Show summary
st.markdown("---")
st.markdown("### 📊 Portfolio Summary")
st.metric("💰 Balance", f"${st.session_state.balance:.2f}")
st.metric("📦 Shares Held", st.session_state.shares)
st.metric("📈 Net Worth", f"${st.session_state.net_worth:.2f}")

# 8. Reset
if st.button("🔁 Reset Simulation"):
    st.session_state.clear()
    st.experimental_rerun()
