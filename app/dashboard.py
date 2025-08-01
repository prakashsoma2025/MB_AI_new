import streamlit as st
import pandas as pd
import pickle
from feature_engineering import generate_features
from model import predict_multibaggers

st.set_page_config(page_title="Multibagger AI - Penny Stock Predictor", layout="wide")

st.title("📈 Multibagger AI: Predict Penny Stocks Under ₹1")
st.markdown("This AI tool identifies potential multibaggers in the penny stock segment (CMP < ₹1) over a 1-year horizon.")

data = pd.read_csv("data/penny_stocks_sample.csv")
model = pickle.load(open("models/model.pkl", "rb"))

features = generate_features(data)
predictions = predict_multibaggers(model, features)

results = data.copy()
results["Prediction"] = predictions["label"]
results["Confidence"] = predictions["confidence"]
results["Target Price (1Y)"] = predictions["target_price"]

st.dataframe(results, use_container_width=True)
