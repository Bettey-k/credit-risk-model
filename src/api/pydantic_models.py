from pydantic import BaseModel
from typing import Optional

class CustomerFeatures(BaseModel):
    Amount: float
    Value: float
    TransactionStartTime: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: int
    ProductId: int
    ProductCategory: int
    ChannelId: int
    PricingStrategy: int
    CustomerId: int

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
