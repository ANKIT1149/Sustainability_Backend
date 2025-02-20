from datetime import timedelta, datetime, timezone
import json
import random
import re
import string
from bson import ObjectId
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
import uuid

from fastapi.encoders import jsonable_encoder
from pymongo import ReturnDocument
import jwt
from fastapi.security import OAuth2PasswordBearer
from backend.utils.send_email_utils import send_email
import bcrypt
from torchvision import datasets
import uvicorn
from backend.utils.ai_model import predict_image
from backend.utils.models import (
    LoginUser,
    RegisterUser,
    UpdateDetail,
    UploadModel,
    VerificationUser,
    WasteModel,
    MessageResponse,
    MessageRequest,
    Product,
    ProductResponse,
    WasteReporter,
)
from backend.utils.db_utils import (
    db,
    fs,
    collections,
    user_collections,
    report_collections,
    ecopoints,
)
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

# secret
SECRET_KEY = "TECHKITINNOVATIVE"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# test
@app.get("/")
def Home():
    return {"message": "Welcome to python backend"}


# image upload
@app.post("/upload", response_model=UploadModel)
async def upload_waste_image(
    file: UploadFile = File(...),
    lat: float = None,
    lon: float = None,
    token: str = Depends(oauth2_scheme),
):
    image_id = str(uuid.uuid4())
    read_file = await file.read()

    image_fs = fs.put(read_file, filename=file.filename)

    token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user = token.get("user_id")

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    image_id_str = user

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


def generate_verification_code():
    return "".join(random.choices(string.digits, k=6))


@app.post("/register")
async def register_user(user: RegisterUser, background: BackgroundTasks):
    if user_collections is None:
        raise HTTPException(status_code=500, detail="Collection not found in db")

    exsistingEmail = user_collections.find_one({"email": user.email})
    if exsistingEmail:
        raise HTTPException(status_code=400, detail="User found with this email")

    hashedpassword = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt())
    verification_code = {
        "code": generate_verification_code(),
        "expiry": datetime.now() + timedelta(minutes=20),
    }

    verification_data = verification_code["code"]
    subject = "OTP Verification(Authentication)"
    body = f"Hello, \n\nWe are from Ecoscan and We are sending OTP for your email verification.\n\nYour Verification code is: {verification_data}.\n\nThanks for conecting us"

    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hashedpassword.decode("utf-8"),
        "is_verified": False,
        "verification_code": verification_code,
    }

    new_users = user_collections.insert_one(user_data)
    background.add_task(send_email, user.email, subject, body)
    return {
        "status": 201,
        "message": "User registered Successfully",
        "user_id": str(new_users.inserted_id),
    }


@app.post("/verify")
async def verify_email(user: VerificationUser):
    exsistingUser = user_collections.find_one({"email": user.email})
    if not exsistingUser:
        raise HTTPException(status_code=404, detail="Email not found")

    verification_data = exsistingUser.get("verification_code", {})
    stored_code = verification_data.get("code", None)
    stored_expiry = verification_data.get("expiry", None)

    if datetime.now() > stored_expiry:
        raise HTTPException(status_code=400, detail="OTP Expired Please Register again")

    if stored_code != user.code:
        raise HTTPException(
            status_code=404, detail="Your verification code nopt amtch.Try agian"
        )

    user_collections.update_one(
        {"_id": exsistingUser["_id"]},
        {
            "$set": {
                "is_verified": True,
                "verification_code": {
                    "code": "",
                    "expiry": datetime.now() + timedelta(seconds=0),
                },
            }
        },
    )

    return {"status": 200, "message": "Email Verified Successfully.Now you can login"}


@app.post("/login")
async def login_user(user: LoginUser):
    user_data = user_collections.find_one({"email": user.email})
    if not user_data:
        raise HTTPException(status_code=404, detail="EMail not found")

    if not user_data["is_verified"]:
        raise HTTPException(status_code=400, detail="User not verified")

    if not bcrypt.checkpw(
        user.password.encode("utf-8"), user_data["password"].encode("utf-8")
    ):
        raise HTTPException(status_code=400, detail="Password not match")

    token_data = {
        "sub": user.email,
        "user_id": str(user_data["_id"]),
        "exp": datetime.now(timezone.utc) + timedelta(days=1),
    }

    token = jwt.encode(token_data, SECRET_KEY, ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "message": "Login successfully",
        "status_code": 200,
    }


@app.post("/waste_report")
async def waste_report(
    token: str = Depends(oauth2_scheme),
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(...),
    location: str = Form(...),
):
    token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = token.get("user_id")

    if not user_id:
        raise HTTPException(status_code=404, detail="User id not found")

    read_file = await file.read()
    image_fs = fs.put(read_file, filename=file.filename)

    ecopints_collection = ecopoints.find_one({"user_id": user_id})
    if not ecopints_collection:
        print("Filter:", {"user_id": user_id})
        print("Update:", {"$inc": {"ecopoints": 10}})

        update_user_ecopoints = {"user_id": user_id, "ecopoints": 10}

        new_ecopoints = ecopoints.insert_one(update_user_ecopoints)

        return {"user_id": user_id, "ecopoints_id": str(new_ecopoints.inserted_id)}

    else:
        updateUser = ecopoints.update_one(
            {"user_id": user_id}, {"$inc": {"ecopoints": 10}}
        )
        updateUser

    user_report = {
        "report_id": user_id,
        "title": title,
        "description": description,
        "location": location,
    }

    new_report = report_collections.insert_one(user_report)

    return {
        "status": 201,
        "message": "Report submitted successfully",
        "user_id": str(new_report.inserted_id),
    }


@app.get("/user_Detail")
async def user_Detail(token: str = Depends(oauth2_scheme)):
    try:
        token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = token.get("user_id")

        if not user_id:
            raise HTTPException(status_code=404, detail="User id not found")

        obj_id = ObjectId(user_id)
        if not obj_id:
            raise HTTPException(status_code=400, detail="Invalid Format")

        user = user_collections.find_one({"_id": obj_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user["_id"] = str(user["_id"])
        return jsonable_encoder(user)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Invalid Signature")


@app.patch("/update_detail")
async def update_detail(user: UpdateDetail, token: str = Depends(oauth2_scheme)):
    token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = token.get("user_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="User id not found")

    obj_id = ObjectId(user_id)
    if not obj_id:
        raise HTTPException(status_code=400, detail="Invalid Id format")

    updated_data = {}

    if user.username:
        updated_data["username"] = user.username

    if user.email:
        updated_data["email"] = user.email

    if user.password and user.password.strip():
        hashedPassword = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt())
        updated_data["password"] = hashedPassword.decode("utf-8")

    updated_user = user_collections.find_one_and_update(
        {"_id": obj_id}, {"$set": updated_data}, return_document=ReturnDocument.AFTER
    )

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    updated_user["_id"] = str(updated_user["_id"])
    data = jsonable_encoder(updated_user)

    return {"status": 200, "message": "User updated successfully", "user": data}


@app.delete("/delete_user")
async def delete_user(token: str = Depends(oauth2_scheme)):

    token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = token.get("user_id")

    if not user_id:
        raise HTTPException(status_code=404, detail="User id not found")

    obj_id = ObjectId(user_id)
    if not obj_id:
        raise HTTPException(status_code=400, detail="Invalid Id format")

    delete_user = user_collections.delete_one({"_id": obj_id})

    return {
        "status": 200,
        "message": "User deleted successfully",
    }


@app.get("/ecopoints")
async def ecopoints_show(token: str = Depends(oauth2_scheme)):
    token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = token.get("user_id")

    if not user_id:
        raise HTTPException(status_code=404, detail="User not found")

    user_ecopoints = ecopoints.find_one({"user_id": user_id})
    if not user_ecopoints:
        raise HTTPException(status_code=404, detail="Collection not found")

    ecopoints_points = user_ecopoints["ecopoints"]
    print(f"Ecopoints earned:", ecopoints_points)

    return {
        "status": 200,
        "message": "User Ecopoints founds",
        "ecopoints": ecopoints_points,
    }


@app.get("/leaderboard_data")
async def leaderboard_data():
    user_data = list(user_collections.find({}, {"_id": 1, "username": 1}))
    print(user_data)

    if not user_data:
        raise HTTPException(status_code=404, detail="User Collection not found")

    ecopoints_data = list(ecopoints.find({}, {"user_id": 1, "ecopoints": 1}))
    print(ecopoints_data)

    if not ecopoints_data:
        raise HTTPException(status_code=404, detail="Ecopoints Collection not found")

    point_dict = {str(p["user_id"]): p["ecopoints"] for p in ecopoints_data}
    leaderboard = []
    for user in user_data:
        user_id = str(user["_id"])
        leaderboard.append(
            {"username": user["username"], "ecopoints": point_dict.get(user_id, 0)}
        )

    leaderboard.sort(key=lambda x: x["ecopoints"], reverse=True)

    return {"leaderboard": leaderboard}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
