from src.modules.base_model import BaseModel
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchcrf import CRF
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from config import (
    LABEL2ID,
    LEARNING_RATE,
    LID2ID,
    WARM_RESTARTS, 
    WEIGHT_DECAY,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
    PADDING
)

class TaskHead(nn.Module):
    def __init__(self, n_labels):
        super(TaskHead, self).__init__()
        self.n_labels = n_labels
        
        self.linear = nn.Sequential(
                nn.Linear(32, n_labels + 1),
                nn.LayerNorm(n_labels + 1)
            )
        self.crf = CRF(
                num_tags=n_labels + 1,
                batch_first=True
            )
    
    def forward(self, x, labels, attention_mask):
        x = self.linear(x)
        loss = -self.crf(x, labels, attention_mask.bool())
        
        path = self.crf.decode(x)
        path = torch.Tensor(path).long()
        
        return loss, path

class SequenceMultiTaskModel(pl.LightningModule):
    def __init__(
        self,
        label2ids, # list of dicts depicting labels to be assigned. Each dict represents a task
        task_names,
        model_name_or_path,
        padding,
        learning_rate,
        weight_decay: float = WEIGHT_DECAY,
        ner_learning_rate: float = LEARNING_RATE,
        lid_learning_rate: float = LEARNING_RATE,
        pos_learning_rate: float = LEARNING_RATE,
        warm_restart_epochs: int = WARM_RESTARTS,
        ner_wd: float = WEIGHT_DECAY,
        lid_wd: float = WEIGHT_DECAY,
        pos_wd: float = WEIGHT_DECAY,
        dropout_rate: float = DROPOUT_RATE
    ) -> None:
        super(SequenceMultiTaskModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.label2ids = label2ids
        self.padding = padding 
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_names = task_names
        
        self.save_hyperparameters()

        self.baseModel = BaseModel(self.model_name_or_path)
        #self.baseModel.freeze()
        
        self.log_vars = []

        # Architecture: base model -> BiLSTM -> CRF
        self.bi_lstm = nn.LSTM(
            input_size=self.baseModel.config.hidden_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        
        self.shared_linear0 = nn.Linear(512, 256)
        self.shared_linear1 = nn.Linear(256, 32)
        #self.shared_linear2 = nn.Linear(128, 64)
        
        self.linear = nn.Sequential(
            self.shared_linear0, 
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            self.shared_linear1,
            nn.LayerNorm(32),
            nn.GELU(),
        )

        self.special_tag_ids = []
        for x in self.label2ids:
            self.special_tag_ids.append(len(x))

        self.task_heads = []

        for i in range(len(self.label2ids)):
            task_id = i

            self.task_heads.append(
                TaskHead( len(self.label2ids[i]) )
            )
            
            if task_names[i] == 'NER':
                self.ner_net = self.task_heads[-1]
            elif task_names[i] == 'LID':
                self.lid_net = self.task_heads[-1]
            else:
                self.pos_net = self.task_heads[-1]
            
            self.log_vars.append(nn.Parameter(torch.zeros(1)))

            self.add_module(f"Task {task_names[task_id]} TaskHead", self.task_heads[task_id])
            self.register_parameter(name=f"Loss param {task_names[task_id]}", param=self.log_vars[task_id])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, task_no = batch
        
        linear_outs = self(self.baseModel(input_ids, attention_mask))
        
        outlist = [ [[], [], []] for x in range(len(self.label2ids)) ]        
        for x in range(len(linear_outs)):
            task_id = task_no[x]
            outlist[task_id][0].append(linear_outs[x])
            outlist[task_id][1].append(attention_mask[x])
            outlist[task_id][2].append(labels[x])
        
        loss = 0
        for task_id in range(len(outlist)):
            if len(outlist[task_id][0]) == 0:
                continue
            
            outlist[task_id] = [ torch.stack(y).to(device) for y in outlist[task_id] ]
            emissions, attention_mask, labels = outlist[task_id]
            outlist[task_id] = self.task_heads[task_id](emissions, labels, attention_mask)
            task_loss, task_path = outlist[task_id]
            loss += torch.exp(-self.log_vars[task_id]) * task_loss + self.log_vars[task_id]
            
            metrics = self._compute_metrics(task_path, labels, f'{self.task_names[task_id]} train', task_id)
            self.log_dict(metrics, on_step=False, on_epoch=True)

        self.log("loss/train", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, task_no = batch
        
        linear_outs = self(self.baseModel(input_ids, attention_mask))
        
        outlist = [ [[], [], []] for x in range(len(self.label2ids)) ]        
        for x in range(len(linear_outs)):
            task_id = task_no[x]
            outlist[task_id][0].append(linear_outs[x])
            outlist[task_id][1].append(attention_mask[x])
            outlist[task_id][2].append(labels[x])
        
        loss = 0
        for task_id in range(len(outlist)):
            if len(outlist[task_id][0]) == 0:
                continue
            
            outlist[task_id] = [ torch.stack(y).to(device) for y in outlist[task_id] ]
            emissions, attention_mask, labels = outlist[task_id]
            outlist[task_id] = self.task_heads[task_id](emissions, labels, attention_mask)
            task_loss, task_path = outlist[task_id]
            loss += task_loss
            
            metrics = self._compute_metrics(task_path, labels, f'{self.task_names[task_id]} val', task_id)
            self.log_dict(metrics, on_step=False, on_epoch=True)

        self.log("loss/val", loss, on_step=False, on_epoch=True)
        return loss

    def forward(self, base_model_outs):
        base_outs = base_model_outs.last_hidden_state
        lstm_outs, _ = self.bi_lstm(base_outs)
        linear_outs = self.linear(lstm_outs)

        return linear_outs
    
    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, mode: str, task_id: int):

        preds = torch.reshape(preds, (-1, 1))
        preds = preds.type_as(labels)

        labels = torch.reshape(labels, (-1, 1))
        
        metrics = {}
        # metrics[f"acc/{mode}"] = accuracy(preds, labels, num_classes=len(self.label2id) + 1, ignore_index=self.special_tag_id)
        metrics[f"prec/{mode}"] = precision(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro", task='multiclass')
        metrics[f"rec/{mode}"] = recall(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro", task='multiclass')
        metrics[f"f1/{mode}"] = f1_score(preds, labels, num_classes=len(self.label2ids[task_id]) + 1, ignore_index=self.special_tag_ids[task_id], average="macro", task='multiclass')

        return metrics
    
    def configure_optimizers(self):
        
        # Same LR for shared params and different LR for different tasks params
        # Same weight decay for shared params and different weight decay for different tasks params 
        # TODO: Experiment with Different LRs
        
        no_decay = ["bias", "LayerNorm.weight"]

        # * The params for which there is no lr or weight_decay key will use global lr and weight_decay 
        # * [ i.e. lr and weight_decay args in AdamW ]
        optimizer_grouped_parameters = [ 
            {
                'params': [
                    p 
                    for n, p in self.bi_lstm.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],

            }, 
            {
                'params': [
                    p 
                    for n, p in self.linear.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
            },
            {
                'params': [
                    p 
                    for n, p in self.ner_net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'lr': self.hparams.ner_learning_rate,
                'weight_decay': self.hparams.ner_wd
            }, 
            {
                'params': [
                    p
                    for n, p in self.lid_net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ], 
                'lr': self.hparams.lid_learning_rate,
                'weight_decay': self.hparams.lid_wd
            },
            {
                'params': [
                    p
                    for n, p in self.pos_net.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ], 
                'lr': self.hparams.pos_learning_rate,
                'weight_decay': self.hparams.pos_wd
            },
            {
                'params': [
                    p 
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ], 
                'weight_decay': 0.0
            }
        ]

        optimizer_grouped_parameters.append({
            'params': [
                p 
                for n, p in self.baseModel.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
        })
        
        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, 
            T_0=self.hparams.warm_restart_epochs,               # First restart after T_0 epochs [50 Initial value, 20 ]
        )

        return [optimizer], [lr_scheduler]