import os
from dotenv import load_dotenv
from fireworks.client import Fireworks

load_dotenv()
client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

response = client.chat.completions.create(
    model="accounts/fireworks/models/gpt-oss-120b",
    messages=[{"role": "user", "content": "Explain RAG in 2 lines"}]
)
print(response.choices[0].message.content)
