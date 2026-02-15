from pydantic import BaseModel


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
    probability: float
    is_high_risk: int
