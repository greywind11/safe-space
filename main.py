from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    user_message: str

@app.post("/chat")
async def chat(msg: Message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a non-judgmental, kind emotional companion. "
                        "You're here to listen, support, and gently respond like a caring friend or a safe diary. "
                        "Avoid giving medical or psychiatric advice. "
                        "Don't act like a therapist. Just be warm, human-like, and emotionally understanding."
                    )
                },
                {
                    "role": "user",
                    "content": msg.user_message
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = response.choices[0].message.content
        return {
            "reply": reply,
            "tokens_used": response.usage.total_tokens,
            "approx_cost": round(response.usage.total_tokens * 0.000005, 4)
        }

    except Exception as e:
        return {"error": str(e)}
