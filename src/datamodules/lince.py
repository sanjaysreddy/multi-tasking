from typing import Optional

import torch 
import pytorch_lightning as pl 

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
from datasets import load_dataset

from config import (
    BATCH_SIZE,
    LABEL2ID,
    LID2ID,
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    PATH_CACHE_DATASET,
    PATH_LINCE_DATASET
)

class LinceDM(pl.LightningDataModule):

    data_map = {
        "lince": {
            "train": [f"{PATH_LINCE_DATASET}/train.json"], 
            "validation": [f"{PATH_LINCE_DATASET}/val.json"]
        }
    }

    def __init__(
        self,
        model_name: str,
        dataset_name: str, 
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING, 
        label2id: dict = LABEL2ID,
        lid2id: dict = LID2ID,
        num_workers: int = NUM_WORKERS,

    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.padding = padding 
        self.label2id = label2id
        self.lid2id = lid2id
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            cache_dir=PATH_BASE_MODELS
        )
    
    def prepare_data(self) -> None:
        load_dataset(
            'json',
            data_files=self.data_map[self.dataset_name], 
            field='data', 
            cache_dir=PATH_CACHE_DATASET
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = load_dataset(
            'json', 
            data_files=self.data_map[self.dataset_name], 
            field="data", 
            cache_dir=PATH_CACHE_DATASET
        )

        self.dataset['train'] = self.dataset['train'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        self.dataset['validation'] = self.dataset['validation'].map(
            self._convert_to_features,
            batched=True,
            drop_last_batch=True,
            batch_size=self.batch_size,
            num_proc=self.num_workers
        )

        self.dataset['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])
        self.dataset['validation'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels', 'lids'])


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset["train"], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset["validation"], 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def _convert_to_features(self, batch, indices=None):
        features = self.tokenizer(
            text=batch['sentence'], 
            max_length=self.max_seq_len,
            padding=self.padding, 
            truncation=True,
            is_split_into_words=True,
        )    

        features["labels"], features["lids"] = self._align_tags(features, batch['bio_tag'], batch["lid"]) 
        return features


    def _align_tags(self, tokenized_outs, tags, lids):
        batch_tags = []
        for example_id in range(0, len(tags)):
            example_tags = []
            currentWord = None
            for word_id in tokenized_outs.word_ids(example_id):

                if (word_id != currentWord):
                    currentWord = word_id 
                    tag = len(self.label2id) if word_id is None else self.label2id[tags[example_id][word_id]]
                    example_tags.append(tag)
                
                elif word_id is None:
                    example_tags.append(len(self.label2id))
                
                else:
                    tag = self.label2id[tags[example_id][word_id]]

                    if tag % 2 == 1:
                        tag += 1
                    
                    example_tags.append(tag)
            
            batch_tags.append(example_tags)
        
        batch_lids = []
        for example_id in range(0, len(lids)):
            example_lids = []
            currentWord = None
            for word_id in tokenized_outs.word_ids(example_id):

                if (word_id != currentWord):
                    currentWord = word_id 
                    lid = len(self.lid2id) if word_id is None else self.lid2id[lids[example_id][word_id]]
                    example_lids.append(lid)
                
                elif word_id is None:
                    example_lids.append(len(self.lid2id))
                
                else:
                    lid = self.lid2id[lids[example_id][word_id]]
                    example_lids.append(lid)
            
            batch_lids.append(example_lids)

        
        return batch_tags, batch_lids
