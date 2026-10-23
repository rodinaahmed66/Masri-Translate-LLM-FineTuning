from transformers import  AutoTokenizer
from configs.config import get_settings

class Base:
    def __init__(self):
        self.setting=get_settings()



class Tokenizer(Base):
    def __init__(self):
        
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.setting.BASE_MODEL_ID)
    
    def tokenize(self,text):
        formated_text=self.tokenizer.apply_chat_template(
            text,
            tokenize=False
        )
        tokens=self.tokenizer(
            formated_text,
            tokenize=True,
            max_length=self.setting.TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
            padding="max_length"
        )
        return tokens
    
    def decode(self,ids):
        response=self.tokenizer.decode(
            ids,
            skip_special_tokens=True
        )
        return response

