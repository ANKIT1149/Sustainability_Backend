# Sustainability_Backend

A smart AI-powered waste detection system that helps users classify waste, provides recycling instructions, finds nearby recycling centers, and suggests eco-friendly alternatives.

# Feature

Upload waste images for AI detection
Get recycling instructions based on waste type
Find the nearest recycling center
Suggest eco-friendly product alternatives
AI chatbot for user queries

# Tech Stack

Frontend: Next.js (React Hook Form)
Backend: FastAPI (Python)
Database: (Mongo Db)
AI Model: FastAI (Trained Model)

🔧 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/ANKIT1149/Sustainability_Backend.git
cd waste-management-ai

2️⃣ Backend Setup (FastAPI)

pip install -r requirements.txt
uvicorn backend.app:app --reload
Note: Ensure you have Python and pip installed.

🚀 API Routes

1️⃣ Upload Waste Image
📌 Endpoint: POST /upload
🔹 Description: Uploads an image and predicts waste type
🔹 Request:
{
"image": "base64 or file"
}
🔹 Response:
{
  "waste_type": "Plastic",
  "recyclable": true,
  "confidence": 98.5
}

2️⃣ Get Recycling Instructions
📌 Endpoint: GET /get_recyclable/{image_id}
🔹 Description: Fetches recycling instructions based on waste type
🔹 Response:
{
  "instructions": "Rinse and place in the plastic recycling bin."
}

3️⃣ Find Nearest Recycling Center
📌 Endpoint: GET /nearest-recycling
🔹 Description: Finds the closest recycling center based on user location
🔹 Response:
{
  "name": "Green Recycle Hub",
  "address": "123 Green Street, City",
}

4️⃣ Suggest Eco-Friendly Alternatives
📌 Endpoint: GET /suggest_alternative/{image_id}
🔹 Description: Suggests sustainable product alternatives
🔹 Response:
{
  "products": [
    {
      "name": "Reusable Cotton Bag",
      "description": "A durable, eco-friendly alternative to plastic bags.",
      "price": "$5.99",
      "product_url": "https://example.com/reusable-bag"
    }
  ]
}

5️⃣ AI Chatbot for Queries
📌 Endpoint: POST /chat
🔹 Description: Users can ask questions related to waste management
🔹 Request:
{
  "message": "How do I recycle glass?"
}
🔹 Response:
{
  "response": "Glass should be rinsed and placed in a glass recycling bin."
}

🎥 Demo Video
📌 (Upload your demo video & link here)

🛠 Future Improvements

1️⃣ Real-time Waste Detection with Live Camera Input
✅ Allow users to scan waste using their mobile camera instead of uploading images.
✅ Implement a live AI model that detects waste instantly.

2️⃣ Gamification & Rewards for Recycling
✅ Introduce a point-based reward system where users earn points for recycling.
✅ Users can redeem points for discounts on eco-friendly products.

3️⃣ Blockchain-Based Waste Tracking
✅ Implement blockchain technology to track waste disposal & recycling history.
✅ Users can see their waste footprint and get incentives for sustainable habits.



📩 Contact
📧 Email: Aryanshraj1139@gmail.com
🔗 Website: [Your Project Link]