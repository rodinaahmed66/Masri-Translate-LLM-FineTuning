import json
import re
import time

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

from configs import get_settings

THINK = re.compile(r"<think.*?>.*?</think\s*>", re.S)  # reasoning models may add this
JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model=None):
        self.settings = get_settings()
        if not self.settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is empty: set it in .env")
        if not (model or self.settings.JUDGE_MODEL):
            raise RuntimeError("JUDGE_MODEL is empty: set it in .env")
        self.client = OpenAI(
            api_key=self.settings.GROQ_API_KEY,
            base_url=self.settings.GROQ_BASE_URL,
            timeout=60.0,
            max_retries=0,
        )
        self.model = model or self.settings.JUDGE_MODEL

    def get_model_name(self):
        return self.model

    def load_model(self):
        return self.client

    @staticmethod
    def construct_prompt(role: str, prompt: str):
        return {"role": role, "content": prompt}

    def generate(self, prompt, schema=None):
        last_error = None
        for attempt in range(self.settings.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[self.construct_prompt("user", prompt)],
                    temperature=self.settings.JUDGE_TEMPERATURE,
                    max_tokens=self.settings.JUDGE_MAX_TOKENS,
                )
                if (
                    not response
                    or not response.choices
                    or not response.choices[0].message
                ):
                    raise RuntimeError("empty response from judge model")
                text = THINK.sub("", response.choices[0].message.content or "").strip()
                if schema is None:
                    return text
                # DeepEval asks for a typed object (score + reason), not text
                found = JSON_BLOCK.search(text)
                return schema(**json.loads(found.group() if found else text))
            except Exception as e:
                last_error = e
                print(
                    f"  judge retry {attempt + 1}/{self.settings.MAX_RETRIES}: "
                    f"{type(e).__name__}: {str(e)[:120]}"
                )
                if attempt < self.settings.MAX_RETRIES - 1:
                    time.sleep(5 * (attempt + 1))  # wait longer after each failure
        raise RuntimeError(
            f"judge failed after {self.settings.MAX_RETRIES} tries"
        ) from last_error

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)
