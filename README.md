# ai_trade
**AI_Trade** is a real-time self-learning stock trading agent powered by Deep Q-Learning. It interacts with live stock market data using the `yfinance` API and makes intelligent Buy/Sell/Hold decisions based on reward signals.

---

##  Features

-  Custom OpenAI Gym-style trading environment  
-  Deep Q-Network (DQN) agent for learning and decision making  
-  Real-time stock/crypto/forex data using `yfinance`  
-  Streamlit-powered interactive UI  
-  Animated UI with trade logging, filtering, and dropdown options  
-  Optional model saving and reloading

ai_trading_agent/
│
├── agent.py              # DQN Agent
├── data_utils.py         # Data preprocessing with indicators (EMA, RSI, MACD)
├── environment.py        # Custom Trading Environment
├── train.py              # Training script for the agent
├── streamlit_app.py      # Frontend dashboard using Streamlit
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation



