from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    famille: str
    year: int
    week_of_year: int
    lag_1: float
    lag_2: float
    lag_4: float
    lag_52: float
    rolling_mean_4: float
    rolling_mean_12: float

class PredictResponse(BaseModel):
    famille: str
    year: int
    week_of_year: int
    predicted_quantity: float

class ModelStatusResponse(BaseModel):
    status: str
    model_name: str
    wmape: float
    r2: float
    last_trained: str
    families_supported: List[str]
