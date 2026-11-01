from transformers import AutoTokenizer

from configs.config import get_settings


class Base:
    def __init__(self):
        self.setting = get_settings()


class Tokenizer(Base):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.setting.BASE_MODEL_ID)

    def tokenize(self, messages, truncation: bool = True):
        """Tokenize a chat conversation (list of role/content dicts)."""
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        kwargs = {"return_tensors": "pt"}
        if truncation:
            kwargs.update(
                truncation=True,
                max_length=self.setting.TOKENIZER_MAX_LENGTH,
            )
        return self.tokenizer(formatted_text, **kwargs)

    def count_tokens(self, messages) -> int:
        """Count tokens after applying the chat template."""
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self.tokenizer.encode(formatted_text, add_special_tokens=False))

    @staticmethod
    def construct_prompt(role: str, prompt: str):
        return {"role": role, "content": prompt}

    def decode(self, ids):
        response = self.tokenizer.decode(ids, skip_special_tokens=True)
        return response
