
# StockFlow AI - inference API (Fitur A & Fitur B).
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = Path("models")
app = FastAPI(title="StockFlow AI Inference API")

restock_model = joblib.load(MODEL_DIR / "restock_predictor_model.joblib")
restock_meta = joblib.load(MODEL_DIR / "restock_predictor_meta.joblib")
discount_model = joblib.load(MODEL_DIR / "discount_demand_response_model.joblib")
discount_meta = joblib.load(MODEL_DIR / "discount_engine_meta.joblib")


class RestockRequest(BaseModel):
    product_id: str
    features: dict  # lag_1, lag_7, ..., stock_akhir, kategori, dll (lihat FEATURES_A_NUM/CAT)


class RestockResponse(BaseModel):
    product_id: str
    prediksi_demand_7hari: float
    rekomendasi_restock: int


@app.post("/restock", response_model=RestockResponse)
def predict_restock(req: RestockRequest):
    cols = restock_meta["features_num"] + restock_meta["features_cat"]
    x = pd.DataFrame([{c: req.features[c] for c in cols}])
    pred = max(0.0, float(restock_model.predict(x)[0]))
    safety_stock = restock_meta["service_level_z"] * restock_meta["residual_std"]
    restock_qty = max(0.0, pred + safety_stock - req.features["stock_akhir"])
    return RestockResponse(
        product_id=req.product_id,
        prediksi_demand_7hari=round(pred, 1),
        rekomendasi_restock=int(np.ceil(restock_qty)),
    )


class DiscountRequest(BaseModel):
    product_id: str
    kategori: str
    stock: float
    cogs: float
    harga_normal: float
    target_days: int
    days_to_expiry_now: int
    asof_date: str
    service_level: float = 0.80


class DiscountResponse(BaseModel):
    rekomendasi_diskon_persen: float
    harga_jual_rekomendasi: int
    probabilitas_habis_dalam_target: float
    ekspektasi_profit_rekomendasi: int


@app.post("/discount", response_model=DiscountResponse)
def predict_discount(req: DiscountRequest):
    # Gunakan recommend_discount() (lihat notebook, Bagian 8.4) dengan discount_model sbg model.
    # Endpoint ini adalah skeleton struktur request/response untuk integrasi FastAPI.
    raise NotImplementedError("Panggil recommend_discount() dari modul training/serving bersama.")
