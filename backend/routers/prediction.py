# from fastapi import APIRouter
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# from preprocessing.data_loader import load_all_data
# from preprocessing.feature_engineering import prepare_features

# router = APIRouter()

# # Load models
# lgb_model = joblib.load("models/lgbm_model.pkl")
# feature_cols = joblib.load("models/lgbm_features.pkl")
# encoders = joblib.load("models/lgbm_encoders.pkl")

# # # Load data once
# data = load_all_data()
# # train_df = data["train"]
# oil_df = data["oil"]
# holidays_df = data["holidays"]
# transactions_df = data["transactions"]
# stores_df = data["stores"]

# # # Preprocess once
# # train_df = prepare_features(train_df, oil_df, holidays_df, transactions_df, stores_df)
# # train_df.dropna(inplace=True)
# train_df = pd.read_parquet("backend/data/preprocessed_train.parquet")

# def safe_transform(encoder, values):
#     values = np.array(values).astype(str)
#     class_to_index = {cls: idx for idx, cls in enumerate(encoder.classes_)}
#     return np.array([class_to_index.get(v, -1) for v in values])

# @router.get("/future")
# def predict_sales(days: int = 30, model: str = "lgbm"):
#     future_start = train_df["date"].max() + pd.Timedelta(days=1)
#     future_dates = pd.date_range(start=future_start, periods=days)

#     all_forecasts = []
#     pairs = train_df[["store_nbr", "family"]].drop_duplicates().head(5)

#     for _, row in pairs.iterrows():
#         store = row["store_nbr"]
#         family = row["family"]

#         hist = train_df[(train_df["store_nbr"] == store) & (train_df["family"] == family)].copy()
#         hist = hist.sort_values("date")

#         # Create empty future rows
#         future_rows = pd.DataFrame({
#             "date": future_dates,
#             "store_nbr": store,
#             "family": family,
#         })

#         # Fill required columns for future rows
#         all_rows = pd.concat([hist, future_rows], ignore_index=True)
#         all_rows = prepare_features(all_rows, oil_df, holidays_df, transactions_df, stores_df)
#         all_rows.sort_values("date", inplace=True)

#         # Encode safely
#         for col, encoder in encoders.items():
#             if col in all_rows.columns:
#                 all_rows[col] = safe_transform(encoder, all_rows[col].astype(str))

#         forecast_rows = all_rows[all_rows["date"].isin(future_dates)]
#         X = forecast_rows[feature_cols]

#         preds = lgb_model.predict(X)

#         all_forecasts.append({
#             "store_nbr": int(store),
#             "family": str(family),
#             "dates": forecast_rows["date"].astype(str).tolist(),
#             "sales": preds.round(2).tolist()
#         })
#         print(all_forecasts)
#     return {
#         "model": model,
#         "forecast": all_forecasts
#     }

from fastapi import APIRouter
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features

router = APIRouter()

# Load model and metadata
lgb_model = joblib.load("models/lgbm_model.pkl")
feature_cols = joblib.load("models/lgbm_features.pkl")
encoders = joblib.load("models/lgbm_encoders.pkl")

# Load static data once
data = load_all_data()
oil_df = data["oil"]
holidays_df = data["holidays"]
transactions_df = data["transactions"]
stores_df = data["stores"]

# Preprocessed training data (includes all columns)
train_df = pd.read_parquet("backend/data/preprocessed_train.parquet")


def safe_transform(encoder, values):
    """Encodes unseen values as -1 safely using label encoder"""
    values = np.array(values).astype(str)
    class_to_index = {cls: idx for idx, cls in enumerate(encoder.classes_)}
    return np.array([class_to_index.get(v, -1) for v in values])


@router.get("/future")
def predict_sales(days: int = 30, model: str = "lgbm"):
    future_start = train_df["date"].max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=days)

    all_forecasts = []
    pairs = train_df[["store_nbr", "family"]].drop_duplicates()

    for _, row in pairs.iterrows():
        store = row["store_nbr"]
        family = row["family"]

        # Historical data for store-family
        hist = train_df[(train_df["store_nbr"] == store) & (train_df["family"] == family)].copy()
        hist = hist.sort_values("date")

        # Create future template
        future_rows = pd.DataFrame({
            "date": future_dates,
            "store_nbr": store,
            "family": family,
        })

        # Combine history and future
        all_rows = pd.concat([hist, future_rows], ignore_index=True)
        all_rows.sort_values("date", inplace=True)
        all_rows.reset_index(drop=True, inplace=True)

        # Prepare features without assuming sales exists
        all_rows = prepare_features(all_rows, oil_df, holidays_df, transactions_df, stores_df)

        # Encode categorical columns
        for col, encoder in encoders.items():
            if col in all_rows.columns:
                all_rows[col] = safe_transform(encoder, all_rows[col].astype(str))

        # Filter only future rows to predict
        forecast_rows = all_rows[all_rows["date"].isin(future_dates)]

        # Prediction
        X = forecast_rows[feature_cols]
        preds = lgb_model.predict(X)

        all_forecasts.append({
            "store_nbr": int(store),
            "family": str(family),
            "dates": forecast_rows["date"].astype(str).tolist(),
            "sales": preds.round(2).tolist()
        })

    return {
        "model": model,
        "forecast": all_forecasts
    }
