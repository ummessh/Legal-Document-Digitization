from langchain.llms.base import LLM
import requests
import json
from typing import Optional, List, Dict, Any

class GroqLLM(LLM):
    def __init__(self, 
                 api_key: str, 
                 model_name: str = "mixtral-8x7b-32768", # Groq's version of Mixtral[Used for it's multilingual capabilities]
                 temperature: float = 0.0,
                 max_tokens: int = 1024):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Create the chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt_template = """
You are an expert in text correction and Named Entity Recognition (NER).
Correct OCR errors in the following text and extract named entities (for English, Marathi, and Hindi).

OCR Text: {ocr_text}

Respond in JSON format:
{
  "corrected_text": "fixed text",
  "entities": [{"entity": "entity_name", "type": "entity_type"}]
}
"""

def process_legal_text(api_key: str, text: str) -> Dict:
    try:
        # Initialize Groq LLM
        llm = GroqLLM(
            api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1  # Low temperature for more consistent outputs
        )
        
        # Create and run chain
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(text=text)
        
        # Parse and validate response
        result = json.loads(response)
        
        # Basic validation
        if not isinstance(result, dict):
            raise ValueError("Invalid response format")
        if "entities" not in result:
            raise ValueError("No entities found in response")
            
        return result
        
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM response"}
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
