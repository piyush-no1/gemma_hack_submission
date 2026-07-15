import json
from typing import Type
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import settings

class LLMClient:
    def __init__(self, model_name: str = settings.OLLAMA_MODEL_JUDGE, temperature: float = 0.1):
        self.model_name = model_name
        self.model = ChatOllama(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature
        )

    def generate_structured(self, system_prompt: str, user_prompt: str, pydantic_schema: Type[BaseModel]) -> BaseModel:
        """
        Invokes local ChatOllama using structured outputs schema binding.
        Includes a robust manual JSON extraction regex fallback in case Olammas fail to bind.
        """
        try:
            # Bind output schema
            structured_llm = self.model.with_structured_output(pydantic_schema)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            result = structured_llm.invoke(messages)
            return result
        except Exception as e:
            print(f"[LLMClient] Structured output binding failed: {e}. Falling back to manual json regex parsing...")
            
            fallback_system_prompt = (
                f"{system_prompt}\n\n"
                f"IMPORTANT: You must respond ONLY with a single valid JSON object matching the JSON schema below.\n"
                f"Do not include conversational filler, markdown formatting (other than JSON blocks), or preamble.\n\n"
                f"JSON Schema:\n{pydantic_schema.model_json_schema()}"
            )
            
            messages = [
                SystemMessage(content=fallback_system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            result = self.model.invoke(messages)
            content = result.content.strip()
            
            try:
                # Extract json code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                content = content.strip()
                data = json.loads(content)
                return pydantic_schema.model_validate(data)
            except Exception as parse_err:
                print(f"[LLMClient] Fallback parsing failed for response: '{content}'. Error: {parse_err}")
                raise parse_err
