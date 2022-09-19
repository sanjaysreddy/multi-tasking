from turtle import forward
import torch
import torch.nn as nn 

from transformers import AutoConfig, AutoModel 

from config import (
    PATH_BASE_MODELS
)

class BaseModel(nn.Module):
    def __init__(
        self, 
        model_name: str,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            cache_dir=PATH_BASE_MODELS, 
        )

        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name, 
            cache_dir=PATH_BASE_MODELS
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor): 
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False   
        
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True 
    

