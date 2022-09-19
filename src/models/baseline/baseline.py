from typing import Optional 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from src.modules.base_model import BaseModel

from config import (
    LABEL2ID,
    LEARNING_RATE,
    LID2ID, 
    WEIGHT_DECAY,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
    PADDING
)

class BaseLine(pl.LightningModule):
    def __init__(
        self, 
        model_name: str, 
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        label2id: dict = LABEL2ID,
        lid2id: dict = LID2ID,
        learning_rate: float = LEARNING_RATE, 
        weight_decay: float = WEIGHT_DECAY,
        dropout_rate: float = DROPOUT_RATE,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.base_model = BaseModel(self.model_name)

    
    def forward(self, *args, **kwargs):
        pass
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pass
    
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer
    
        