import pandas as pd

def generate_features(df):
    df_feat = df.copy()
    df_feat["Price Momentum 1M"] = (df["price"] - df["price"].shift(20)) / df["price"].shift(20)
    df_feat["Price Momentum 3M"] = (df["price"] - df["price"].shift(60)) / df["price"].shift(60)
    df_feat["Volume Change"] = df["volume"] / df["avg_volume"]
    df_feat = df_feat.fillna(0)
    return df_feat[["Price Momentum 1M", "Price Momentum 3M", "Volume Change"]]