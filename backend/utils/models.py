from pydantic import BaseModel
from typing import List, Optional


class WasteModel(BaseModel):
    image_id: str
    waste_type: str
    recyclable: bool
    confidence: Optional[float]


class UploadModel(BaseModel):
    message: str
    data: WasteModel


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str


class Product(BaseModel):
    name: str
    price: float
    purchase_link: str
    description: str


class ProductResponse(BaseModel):
    waste_type: str
    recommended_products: List[Product]
