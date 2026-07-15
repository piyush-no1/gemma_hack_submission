import json
from typing import Type
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import settings

class LLMClient:
    def __init__(self, model_name: str = settings.OLLAMA_MODEL_JUDGE, temperature: float = 0.1):
        self.model_name = model_name
        self.api_key = settings.get_google_api_key()
        
        # Route all model requests to Google GenAI API if an API key is present
        self.use_google = bool(self.api_key)
        
        if self.use_google:
            from google import genai
            # Map all local/custom models to the official Google Gemma 4 (31B) model
            self.google_model_name = "gemma-4-31b-it"
            self.client = genai.Client(api_key=self.api_key)
            print(f"[LLMClient] Routed model '{self.model_name}' to Google GenAI Cloud API as '{self.google_model_name}'.")
        else:
            self.model = ChatOllama(
                model=model_name,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=temperature
            )

    def generate_structured(self, system_prompt: str, user_prompt: str, pydantic_schema: Type[BaseModel]) -> BaseModel:
        """
        Invokes model using structured outputs schema binding.
        Supports both local Ollama and Google GenAI API depending on configuration.
        """
        if self.use_google:
            print(f"[LLMClient] Querying model '{self.google_model_name}' on Google GenAI Cloud API. Please wait...")
            from google.genai import types
            content = ""
            try:
                response = self.client.models.generate_content(
                    model=self.google_model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=pydantic_schema,
                    ),
                )
                content = response.text.strip()
                parsed = pydantic_schema.model_validate_json(content)
                print(f"[LLMClient] successfully parsed structured schema from Google GenAI model '{self.google_model_name}'.")
                return parsed
            except Exception as e:
                print(f"[LLMClient] Google GenAI structured query failed: {e}. Falling back to manual JSON parse...")
                try:
                    response = self.client.models.generate_content(
                        model=self.google_model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt + f"\n\nRespond ONLY with a valid JSON object matching the schema: {json.dumps(pydantic_schema.model_json_schema())}",
                            temperature=0.1,
                        ),
                    )
                    content = response.text.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    content = content.strip()
                    parsed = pydantic_schema.model_validate_json(content)
                    print(f"[LLMClient] successfully parsed raw JSON fallback schema from Google GenAI model '{self.google_model_name}'.")
                    return parsed
                except Exception as parse_err:
                    print(f"[LLMClient] Google GenAI fallback parsing failed: {parse_err}")
                    raise ValueError(
                        f"LLMClient: Google GenAI structured output and JSON fallback failed for model '{self.google_model_name}'. "
                        f"Raw response: '{content[:200]}'"
                    ) from parse_err

        print(f"[LLMClient] Querying model '{self.model_name}' (Ollama URL: {settings.OLLAMA_BASE_URL}). Please wait...")
        try:
            # Bind output schema
            structured_llm = self.model.with_structured_output(pydantic_schema)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            result = structured_llm.invoke(messages)
            print(f"[LLMClient] successfully parsed structured schema for model '{self.model_name}'.")
            return result
        except Exception as e:
            print(f"[LLMClient] Structured output binding failed: {e}. Falling back to manual json regex parsing for '{self.model_name}'...")
            
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
            
            print(f"[LLMClient] Sending raw fallback prompt to model '{self.model_name}'...")
            result = self.model.invoke(messages)
            content = result.content.strip()
            print(f"[LLMClient] Received raw text response from model '{self.model_name}'. Length: {len(content)} characters. Parsing...")
            
            try:
                # Extract json code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                content = content.strip()
                data = json.loads(content)
                parsed_model = pydantic_schema.model_validate(data)
                print(f"[LLMClient] successfully parsed raw JSON fallback schema for model '{self.model_name}'.")
                return parsed_model
            except Exception as parse_err:
                print(f"[LLMClient] Fallback parsing failed for response: '{content[:200]}'. Error: {parse_err}")
                raise ValueError(
                    f"LLMClient: both structured output and JSON fallback failed for model '{self.model_name}'. "
                    f"Raw response: '{content[:200]}'"
                ) from parse_err
