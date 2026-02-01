import openai
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class LLMPrompting:
def init(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
self.model_name = model_name
if api_key is None:
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
def build_prompt(self, context: str, question: str) -> str:
    """Build the prompt for the LLM."""
    prompt = f"""You are a strict document-based AI assistant.
    def get_answer(self, prompt: str) -> str:
    """Get answer from the LLM."""
    response = openai.ChatCompletion.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    return response.choices[0].message.content