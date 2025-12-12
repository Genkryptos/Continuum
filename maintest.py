from dotenv import load_dotenv
from LLMManager import LocalOpenAI

load_dotenv()
client = LocalOpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say REAL if you are not dummy."}],
)
print(resp.choices[0].message.content)