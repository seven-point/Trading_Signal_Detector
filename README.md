# 📈 Stock Sentiment Analyzer

A lightweight **Stock Sentiment Analyzer** that uses real-time news headlines to provide sentiment insights for any stock ticker. Built with **Streamlit**, it fetches financial news via the **Marketaux API** and uses a **custom rule-based logic** to classify sentiments.

---

## 🚀 Features

- 🔍 Input any stock ticker (e.g., `AAPL`, `TSLA`, `NVDA`)
- 📰 Fetches latest news headlines using Marketaux
- 🧠 Analyzes headline sentiment (Positive / Neutral / Negative)
- 📊 Displays sentiment breakdown as a pie chart
- 📌 Actionable recommendation: **Buy / Hold / Sell**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI Framework
- **Marketaux API** – News Provider
- **Custom Sentiment Engine** – No external models like FinBERT used

---

## 📦 Installation

1. **Clone the repository**
   bash
   git clone https://github.com/yourusername/stock-sentiment-analyzer.git
   cd stock-sentiment-analyzer

2. **Create a virtual environment (optional but recommended)**

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**

bash
pip install -r requirements.txt
Set up API Key

#Create a .env file in the project root and add your Marketaux key:
MARKETAUX_API_KEY=your_api_key_here

4. **Running the App**
bash
streamlit run app.py

##**How It Works:**
User enters a valid stock ticker (e.g., NFLX)

News headlines are fetched via Marketaux API

A basic rule-based sentiment engine assigns sentiment to each headline:

Keywords like "soars", "beats", "gains" → Positive

Words like "falls", "lawsuit", "loss" → Negative

Others → Neutral

Pie chart displays the overall distribution

Recommendation logic:

✅ ≥ 60% Positive → BUY

❌ ≥ 60% Negative → SELL

🔄 Otherwise → HOLD
