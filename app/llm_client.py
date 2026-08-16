import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache

from groq import Groq
from pydantic import BaseModel, Field


class SqlGenerationResult(BaseModel):
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Plain-English explanation of what the query does")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's self-reported confidence, 0 to 1")
    tables_used: list[str] = Field(description="Tables referenced in the query")


class BackTranslationResult(BaseModel):
    question: str = Field(description="The natural-language question this SQL query answers")


class LLMClient(ABC):
    @abstractmethod
    def generate_sql(self, prompt: str) -> SqlGenerationResult: ...

    @abstractmethod
    def back_translate_sql(self, prompt: str) -> BackTranslationResult: ...


class GroqLLMClient(LLMClient):
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a key at https://console.groq.com/keys"
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def _structured_call(self, prompt: str, schema_model: type[BaseModel]):
        schema = schema_model.model_json_schema()
        schema["additionalProperties"] = False

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__.lower(),
                    "strict": True,
                    "schema": schema,
                },
            },
        )

        data = json.loads(response.choices[0].message.content)
        return schema_model.model_validate(data)

    def generate_sql(self, prompt: str) -> SqlGenerationResult:
        return self._structured_call(prompt, SqlGenerationResult)

    def back_translate_sql(self, prompt: str) -> BackTranslationResult:
        return self._structured_call(prompt, BackTranslationResult)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Cached so each request reuses one client — and so a missing API key
    raises inside a request handler rather than at import time."""
    return GroqLLMClient()