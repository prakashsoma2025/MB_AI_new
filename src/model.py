import pandas as pd

def predict_multibaggers(model, features_df):
    pred = model.predict(features_df)
    conf = model.predict_proba(features_df)[:, 1]
    target_price = features_df.index.to_series().apply(lambda x: round((1 + conf[x]) * 1, 2))  # Dummy logic
    return pd.DataFrame({
        "label": ["Multibagger" if p == 1 else "No" for p in pred],
        "confidence": (conf * 100).round(2).astype(str) + "%",
        "target_price": target_price
    })