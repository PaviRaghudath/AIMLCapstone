import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import lightgbm as lgb

from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features





def safe_label_encode(encoder, values):
    known_classes = set(encoder.classes_)
    return [encoder.transform([v])[0] if v in known_classes else -1 for v in values]

def predict_sales(fastApiRequest, model_type="lgbm", input_df: pd.DataFrame = None) -> pd.DataFrame:

    lgb_model = fastApiRequest.app.state.lgb_model
    # joblib.load("models/lgbm_model.pkl")
    # nn_model = load_model("models/nn_model.keras")
    family_encoder = fastApiRequest.app.state.family_encoder
    # joblib.load("models/family_encoder.pkl")
    # scaler = joblib.load("models/scaler.pkl")
    features = fastApiRequest.app.state.lgbm_features
    # joblib.load("models/lgbm_features.pkl")
    city_encoder = fastApiRequest.app.state.city_encoder
    # joblib.load("models/city_encoder.pkl")
    state_encoder = fastApiRequest.app.state.state_encoder
    # joblib.load("models/state_encoder.pkl")
    type_encoder = fastApiRequest.app.state.type_encoder
    # joblib.load("models/type_encoder.pkl")

    if input_df is None:
        raise ValueError("You must provide an input DataFrame.")

    
    data = fastApiRequest.app.state.raw_df
    oil = data["oil"]
    holidays = data["holidays"]
    transactions = data["transactions"]
    stores = data["stores"]
    train_df = data["train"]  

    
    
    latest_date = input_df["date"].max()
    min_history_date = latest_date - pd.Timedelta(days=35)  
    train_df = train_df[train_df["date"] >= min_history_date]

  
    train_df["id"] = train_df["store_nbr"].astype(str) + "_" + train_df["family"].astype(str)
    input_df["id"] = input_df["store_nbr"].astype(str) + "_" + input_df["family"].astype(str)

    combined_df = pd.concat([train_df.copy(), input_df.copy()], ignore_index=True)

    
    enriched_df = prepare_features(combined_df, oil, holidays, transactions, stores)

    
    final_df = enriched_df[enriched_df["date"] == latest_date].copy()

    if final_df.empty:
        print("No rows left after feature preparation (probably due to missing lag data).")
        input_df["predicted_sales"] = 50.0
        return input_df

    final_df["family"] = safe_label_encode(family_encoder, final_df["family"])
    final_df["city"] = safe_label_encode(city_encoder, final_df["city"])
    final_df["state"] = safe_label_encode(state_encoder, final_df["state"])
    final_df["type"] = safe_label_encode(type_encoder, final_df["type"])


    final_df = final_df[
    (final_df["family"] != -1) &
    (final_df["city"] != -1) &
    (final_df["state"] != -1) &
    (final_df["type"] != -1)
]

    # features = joblib.load("models/lgbm_features.pkl")
    X = final_df[features]

    preds = lgb_model.predict(X)
    final_df["predicted_sales"] = preds
    if np.issubdtype(final_df["family"].dtype, np.integer):
        final_df["family"] = family_encoder.inverse_transform(final_df["family"].astype(int))

    return final_df[["store_nbr", "family", "predicted_sales"]]
