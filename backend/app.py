import random
import re
from fastapi import FastAPI, File, HTTPException, UploadFile
import uuid

import uvicorn
from backend.utils.ai_model import predict_image
from backend.utils.models import (
    UploadModel,
    WasteModel,
    MessageResponse,
    MessageRequest,
    Product,
    ProductResponse,
)
from backend.utils.db_utils import db, fs, collections
from backend.utils.deep_ai import anylyze_image
from backend.utils.openai import generate_answer_from_waste
from backend.utils.getloation import find_nearest_recycling_center
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import os
from PIL import Image

app = FastAPI()

# env
dotenv_path = "backend/.env"
load_dotenv(dotenv_path=dotenv_path)

# openai
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
openai.api_key = OPENAI_API_KEY

# cors
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# test
@app.get("/")
def Home():
    return {"message": "Welcome to python backend"}


# image upload
@app.post("/upload", response_model=UploadModel)
async def upload_waste_image(
    file: UploadFile = File(...), lat: float = None, lon: float = None
):
    image_id = str(uuid.uuid4())
    read_file = await file.read()

    image_fs = fs.put(read_file, filename=file.filename)

    image_id_str = str(image_fs)

    final_prediction = anylyze_image(read_file)

    prediction_result = WasteModel(
        image_id=image_id_str,
        waste_type=final_prediction["waste_type"],
        recyclable=final_prediction["recyclable"],
        confidence=final_prediction["confidence"],
    )

    collections.insert_one(prediction_result.model_dump())
    return {"message": "Image upload and analyzed...", "data": prediction_result}


# give instruction basic of waste_type
@app.get("/get_recyclable/{image_id}")
async def get_recyclable_input(image_id: str):
    waste_data = collections.find_one({"image_id": image_id})

    if not waste_data:
        return {"error": "Image not found"}

    waste_types = waste_data["waste_type"]

    instruction = generate_answer_from_waste(waste_types)

    return {"waste_type": waste_types, "instruction": instruction}


# get the nearest recycling centre location
@app.get("/nearest-recycling")
async def nearest_recycling(lat: float, lon: float):
    return find_nearest_recycling_center(lat, lon)


# ai cahtbot
@app.post("/chat", response_model=MessageResponse)
async def chat_bot(request: MessageRequest):
    user_message = request.message

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=150,
            temperature=0.7,
        )

        bot_message = response.choices[0].message["content"].strip()
        return MessageResponse(response=bot_message)

    except Exception as e:
        return MessageResponse(response=f"Error: {str(e)}")


# suggest alternative product to prevnet ecosystem
@app.get("/suggest_alternative/{image_id}", response_model=ProductResponse)
async def suggest_alternative(image_id: str):
    prediction = collections.find_one({"image_id": image_id})

    if not prediction:
        raise HTTPException(
            status_code=404, detail="No waste type found for this image"
        )

    waste_type = prediction["waste_type"]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in eco-friendly product recommendations.",
                },
                {
                    "role": "user",
                    "content": f"Suggest eco-friendly alternatives to replace {waste_type} items.Also suggest online place to buy alternative product",
                },
            ],
            max_tokens=150,
            temperature=0.7,
        )

        ai_response = response["choices"][0]["message"]["content"].strip()

        suggestions = []
        for line in ai_response.split("\n"):
            line = line.strip()
            clean_name = re.sub(r"^\d+\.\s*", "", line)

            if clean_name:
                suggestions.append(clean_name)

        recommended_products = [
            {
                "name": suggestion,
                "description": f"This product can help you reduce your usage of {waste_type}.",
                "price": round(random.uniform(5.0, 50.0), 2),
                "purchase_link": f"https://example.com/product/{re.sub(r'[^a-zA-Z0-9_]', '_', suggestion)}_{uuid.uuid4().hex[:6]}",
            }
            for suggestion in suggestions
        ]

        return {"waste_type": waste_type, "recommended_products": recommended_products}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)



