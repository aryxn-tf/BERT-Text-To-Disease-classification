import torch
import transformers

class TrainDataset:
    def __init__(self,texts,targets,tokenizer_path,max_len=64):
        self.texts = texts
        self.targets = targets
        self.tokenizer_path = tokenizer_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        target = self.targets[idx]
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens = True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )

        return {
            "ids": torch.tensor(inputs['input_ids'],dtype=torch.long),
            "mask": torch.tensor(inputs['attention_mask'],dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"],dtype=torch.long),
            "targets": torch.tensor(target,dtype=torch.long)
        }
    

class TestDataset:
    def __init__(self,texts,tokenizer_path,max_len=64):
        self.texts = texts
        self.tokenizer_path = tokenizer_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens = True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True
        )

        return {
            "ids": torch.tensor(inputs['input_ids'],dtype=torch.long),
            "mask": torch.tensor(inputs['attention_mask'],dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"],dtype=torch.long)
        }