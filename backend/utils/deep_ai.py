import requests
import os
from backend.utils.ai_model import predict_image
from dotenv import load_dotenv

dotenv_path = "backend/.env"
load_dotenv(dotenv_path=dotenv_path)

DEEPAI_API_KEY = os.getenv("DEEP_API_KEY")
print(DEEPAI_API_KEY)


# anylyze image by both ai to get better response
def anylyze_image(image_bytes):
    try:
        custom_model_prediction = predict_image(image_bytes)
        custom_model_confidence = custom_model_prediction.get("confidence", 0)
    except Exception as e:
        print("Your AI model failed", e)
        custom_model_prediction = None
        custom_model_confidence

    deepai_prediction = None

    try:
        url = "https://api.deepai.org/api/content-moderation"
        response = requests.post(
            url, files={"image": image_bytes}, headers={"api-key": DEEPAI_API_KEY}
        )

        deepAI_result = response.json()
        print("Deep ai response:", deepAI_result)

        if "output" in deepAI_result:
            deepai_prediction = {
                "waste_type": (
                    deepAI_result["output"][0] if deepAI_result["output"] else "unknown"
                ),
                "confidence": 0.5,
                "recyclable": deepAI_result["output"][0]
                in ["newspaper", "magazines", "cardboard_boxes", "aluminum_cans"],
            }

            print(deepai_prediction)
        else:
            deepai_prediction = None
    except Exception as e:
        print("DeepAI Failed:", e)

    if custom_model_prediction and (
        custom_model_confidence >= 0.5 or not deepai_prediction
    ):
        return custom_model_prediction
    elif deepai_prediction:
        return deepai_prediction
    else:
        return {"waste_type": "unknown", "recyclable": False, "confidence": 0}
