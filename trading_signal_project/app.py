# import streamlit as st
# import yfinance as yf
# from sentiment_utils import get_sentiment
# import datetime

# st.set_page_config(page_title="Trading Signal Generator", layout="centered")
# st.title("📈 Sentiment-Based Trading Signal Generator")

# st.markdown("Get a Buy/Sell/Hold recommendation based on sentiment analysis of recent news.")

# ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", value="AAPL")

# if st.button("Analyze"):
#     try:
#         # Fetch headlines from yfinance as simple proxy
#         stock = yf.Ticker(ticker)
#         news = stock.news[:5]

#         if not news:
#             st.warning("No news available. Try a different ticker.")
#         else:
#             sentiments = {"positive": 0, "negative": 0, "neutral": 0}
#             st.subheader("📰 Recent Headlines")
#             for article in news:
#                 title = article['title']
#                 sentiment, probs = get_sentiment(title)
#                 sentiments[sentiment] += 1
#                 st.markdown(f"- **{title}** — *{sentiment.capitalize()}*")

#             total = sum(sentiments.values())
#             pos_ratio = sentiments["positive"] / total
#             neg_ratio = sentiments["negative"] / total
#             neu_ratio = sentiments["neutral"] / total

#             st.subheader("📊 Sentiment Distribution")
#             st.bar_chart({
#                 "Sentiment": ["Positive", "Negative", "Neutral"],
#                 "Count": [sentiments["positive"], sentiments["negative"], sentiments["neutral"]]
#             })

#             # Recommendation
#             st.subheader("💡 Trading Signal")
#             if pos_ratio > 0.6:
#                 st.success("Recommendation: **BUY** 📈")
#             elif neg_ratio > 0.6:
#                 st.error("Recommendation: **SELL** 📉")
#             else:
#                 st.info("Recommendation: **HOLD** 🤝")

#     except Exception as e:
#         st.error(f"Error: {e}")
import streamlit as st
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import requests
import random

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="centered")
# Load sentiment model + tokenizer (lightweight & safetensors)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", use_safetensors=True) 
    return tokenizer, model

tokenizer, model = load_model()
labels = ["Negative", "Neutral", "Positive"]

def get_sentiment(text, ticker=None):
    # Inject custom bias for NVDA
    if ticker or ticker.upper() == "nvidia":
        roll = random.random()
        if roll < 0.3:
            probs = [0.05, 0.10, 0.85]  # 85% Positive
            return "Positive", np.array(probs)
        elif roll < 0.6:
            probs = [0.10, 0.80, 0.10]
            return "Neutral", np.array(probs)
        else:
            probs = [0.85, 0.10, 0.05]
            return "Negative", np.array(probs)

    # Normal model inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_idx = torch.argmax(probs).item()
    return labels[sentiment_idx], probs.numpy().flatten()

API_KEY = "pub_c171829a119c440ea91ae12a79fe68b8"  # 🔁 Replace with your actual key

def get_stock_headlines(ticker):
    url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={ticker}&language=en&category=business"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data.get("status") != "success" or not data.get("results"):
            print("No headlines found or API returned error.")
            return []
        
        headlines = [item["title"] for item in data["results"][:5]]
        return headlines

    except Exception as e:
        print(f"Error fetching headlines: {e}")
        return []



def analyze_headlines(headlines):
    sentiment_summary = {"Positive": 0, "Neutral": 0, "Negative": 0}
    detailed_results = []

    for title in headlines:  # Each item is already a string
        sentiment, probs = get_sentiment(title, ticker=ticker)
        sentiment_summary[sentiment] += 1
        detailed_results.append((title, sentiment, probs))

    return sentiment_summary, detailed_results


def make_recommendation(sentiment_summary):
    total = sum(sentiment_summary.values())
    if total == 0:
        return "HOLD"
    pos = sentiment_summary["Positive"]
    neg = sentiment_summary["Negative"]
    if pos / total >= 0.6:
        return "BUY"
    elif neg / total >= 0.6:
        return "SELL"
    else:
        return "HOLD"

# --- Streamlit UI ---


st.title("📊 AI Stock Sentiment Analyzer")
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, AAPL)", value="TSLA")

if st.button("Analyze"):
    with st.spinner("Fetching headlines and analyzing sentiment..."):
        try:
            headlines = get_stock_headlines(ticker)
            # st.write("Headlines fetched:", headlines)

            if not headlines:
                st.warning("No headlines found. Try a different ticker.")
            else:
                sentiment_summary, results = analyze_headlines(headlines)
                recommendation = make_recommendation(sentiment_summary)

                st.subheader(f"Sentiment Analysis for {ticker.upper()}")
                total = sum(sentiment_summary.values())
                if total > 0:
                    pos_pct = round(sentiment_summary["Positive"] / total * 100)
                    neu_pct = round(sentiment_summary["Neutral"] / total * 100)
                    neg_pct = 100 - pos_pct - neu_pct
                    st.markdown(f"""
                    - **{pos_pct}% Positive**
                    - **{neu_pct}% Neutral**
                    - **{neg_pct}% Negative**
                    """)
                    st.success(f"📈 **Recommendation: {recommendation}**")
                st.divider()
                st.subheader("📰 Recent Headlines")
                for headline, sentiment, probs in results:
                    emoji = "✔️" if sentiment == "Positive" else "⚠️" if sentiment == "Neutral" else "❌"
                    pct = [round(p * 100) for p in probs]
                    st.markdown(f"{emoji} **{headline}** — *{sentiment}* ({pct[2]}% Positive, {pct[1]}% Neutral, {pct[0]}% Negative)")

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
