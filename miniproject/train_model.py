import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


data = pd.read_csv("dataset/hotel_reviews.csv", encoding="latin1")

data["Review"] = data["Review"].apply(clean_text)


def convert_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"


data["Sentiment"] = data["Rating"].apply(convert_sentiment)


X = data["Review"]
y = data["Sentiment"]


vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


pickle.dump(model, open("sentiment/model.pkl", "wb"))
pickle.dump(vectorizer, open("sentiment/vectorizer.pkl", "wb"))

print("Model trained successfully")