# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np

# # Load model and tokenizer once
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# def get_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     sentiment_idx = torch.argmax(probs).item()
#     sentiment_label = ["positive", "negative", "neutral"][sentiment_idx]
#     return sentiment_label, probs.numpy().flatten()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment(headlines):
    sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
    labeled = []

    for text in headlines:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()

        # 1-2 stars: Negative, 3 stars: Neutral, 4-5 stars: Positive
        if pred <= 1:
            label = "Negative"
            sentiments["Negative"] += 1
        elif pred == 2:
            label = "Neutral"
            sentiments["Neutral"] += 1
        else:
            label = "Positive"
            sentiments["Positive"] += 1

        labeled.append((text, label))

    total = sum(sentiments.values())
    percent_positive = round(sentiments["Positive"] / total * 100, 1)
    percent_neutral = round(sentiments["Neutral"] / total * 100, 1)
    percent_negative = round(sentiments["Negative"] / total * 100, 1)

    recommendation = "BUY" if percent_positive >= 60 else "HOLD" if percent_positive >= 40 else "SELL"

    return {
        "percent_positive": percent_positive,
        "percent_neutral": percent_neutral,
        "percent_negative": percent_negative,
        "recommendation": recommendation,
        "labeled": labeled,
    }
