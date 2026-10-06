import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from google import genai
from google.genai import types
from openai import AsyncOpenAI

from logger import analysis_log, app_log


class BaseLLMService(ABC):
    """Base class for LLM services"""

    @abstractmethod
    async def generate_response(
        self, messages: List[Dict[str, str]], json_mode: bool = False, **kwargs
    ) -> str:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    async def generate_vision(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a response from the LLM for vision tasks"""
        pass


class OpenAIService(BaseLLMService):
    """OpenAI implementation of the LLM service"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_response(
        self, messages: List[Dict[str, str]], json_mode: bool = False, **kwargs
    ) -> str:
        model = kwargs.pop("model", "gpt-4o-mini")
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def generate_vision(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=4096, **kwargs
        )
        return response.choices[0].message.content


class GeminiService(BaseLLMService):
    """Google Gemini implementation of the LLM service"""

    MODEL = "gemini-2.5-flash"

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _split_messages(messages: List[Dict[str, Any]]):
        """Convert OpenAI-style messages into a Gemini system instruction + parts"""
        system_prompt = "\n".join(
            msg["content"] for msg in messages if msg["role"] == "system"
        )
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            content = msg["content"]
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
                continue
            for item in content:
                if item["type"] == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        mime_type = url.split(";")[0].split(":")[1]
                        data = base64.b64decode(url.split("base64,")[1])
                        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        return system_prompt or None, parts

    async def _generate(
        self, messages: List[Dict[str, Any]], json_mode: bool = False
    ) -> str:
        system_prompt, parts = self._split_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Thinking adds a lot of latency and isn't needed for OCR or scoring
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if json_mode else None,
        )
        response = await self.client.aio.models.generate_content(
            model=self.MODEL, contents=parts, config=config
        )
        return response.text

    async def generate_response(
        self, messages: List[Dict[str, str]], json_mode: bool = False, **kwargs
    ) -> str:
        return await self._generate(messages, json_mode=json_mode)

    async def generate_vision(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        return await self._generate(messages)


def get_llm_service() -> BaseLLMService:
    """Factory function to get the appropriate LLM service based on available API keys"""

    provider = os.getenv("PROVIDER")

    if provider == "openai":
        return OpenAIService()
    elif provider == "google":
        return GeminiService()
    else:
        raise ValueError(
            "No API keys found. Please set either OPENAI_API_KEY or GOOGLE_API_KEY in your .env file"
        )
