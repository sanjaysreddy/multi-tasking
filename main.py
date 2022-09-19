import argparse

import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.models.baseline.baseline import BaseLine
from src.datamodules.lince import LinceDM

from config import (
    MAX_EPOCHS,
    LEARNING_RATE,
    PATH_EXPERIMENTS,
    PROJECT_NAME,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
    PADDING,
    BATCH_SIZE,
    BASE_MODEL,
    NUM_WORKERS,
    AVAIL_GPUS,
)

def test_dm(args):
    dm = LinceDM(
        model_name=args.base_model, 
        dataset_name=args.dataset, 
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        num_workers=args.workers
    )

    dm.setup()
    print(next(iter(dm.train_dataloader())))


def main(args):
    # Init DM 

    # Init Model 

    # Init Logger & Trainer 
    
    logger = WandbLogger(
        name="", 
        save_dir=PATH_EXPERIMENTS,
        id="",
        project=PROJECT_NAME,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        logger=logger,
    )

    # Runs



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Hyperparams
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set max epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Set Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Set Weight Decay")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Set dropout rate")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQUENCE_LENGTH, help="Set max seq length")
    parser.add_argument("--padding", type=str, default=PADDING, help="Set padding style")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL, help="Set base transformer model")
    parser.add_argument("--unfreeze", type=bool, default=None, help="Freeze or Unfreeze base model")
    parser.add_argument("--dataset", type=str, default="lince", help="Set dataset to be used")

    # Hardware
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Set CPU Threads")
    parser.add_argument("--gpus", type=int, default=AVAIL_GPUS, help="Set no. of GPUs required")

    args = parser.parse_args()

    test_dm(args)