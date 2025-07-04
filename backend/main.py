from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import prediction, rag, charts
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.routers.charts import router as charts_router
import joblib
from contextlib import asynccontextmanager
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load raw data
    raw_df = load_all_data()
    oil = raw_df["oil"]
    holidays = raw_df["holidays"]
    transactions = raw_df["transactions"]
    stores = raw_df["stores"]
    train_df = raw_df["train"]  

    # Generate features
    features_df = prepare_features(train_df, oil, holidays, transactions, stores)

    # Load model and encoders
    lgb_model = joblib.load("models/lgbm_model.pkl")
    # nn_model = load_model("models/nn_model.keras")
    family_encoder = joblib.load("models/family_encoder.pkl")
    # scaler = joblib.load("models/scaler.pkl")
    lgbm_features = joblib.load("models/lgbm_features.pkl")
    city_encoder = joblib.load("models/city_encoder.pkl")
    state_encoder = joblib.load("models/state_encoder.pkl")
    type_encoder = joblib.load("models/type_encoder.pkl")

    # Store them in app state
    app.state.raw_df = raw_df
    # app.state.train_df = train_df
    app.state.features_df = features_df
    app.state.lgb_model = lgb_model
    app.state.family_encoder = family_encoder
    app.state.lgbm_features = lgbm_features
    app.state.city_encoder = city_encoder
    app.state.state_encoder = state_encoder
    app.state.type_encoder = type_encoder

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router, prefix="/predict")
app.include_router(rag.router, prefix="/rag")
app.include_router(charts_router)