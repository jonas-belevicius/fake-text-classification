"""
This module trains a DistilBERT model on the fake news dataset.

The module uses the PyTorch Lightning library to train the model. The training
process is defined in the `train_model` function. The function takes the model,
data module, number of epochs, learning rate, and other parameters as input
and trains the model on the data.

"""

import os
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

from config import (
    TRAIN_EPOCHS,
    TRAIN_LR,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    LOGS_PATH,
    LOGS_NAME,
    RANDOM_SEED,
    DROPOUT,
    MAX_SEQ_LEN,
    WEIGHT_DECAY,
    DATA_DIR,
)
from pytorch_modules import FakeTextDataModule, FakeTextModule
from custom_functions import train_model

if __name__ == "__main__":

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased",
    )
    classifier_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
    )

    train_data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "train_processed_pipeline_five.csv",
        )
    )
    val_data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "val_processed_pipeline_five.csv",
        )
    )
    test_data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "test_processed_pipeline_five.csv",
        )
    )

    print(
        f"Batch size: {BATCH_SIZE}",
        f"Train lr: {TRAIN_LR}",
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_accuracy", patience=3, verbose=True, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename="best_checkpoint-fifth-model-{MAX_SEQ_LEN}-{epoch}-{val_accuracy}",
    )

    logger = CSVLogger(
        LOGS_PATH,
        name=LOGS_NAME,
        flush_logs_every_n_steps=10,
    )

    text_data_module = FakeTextDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    text_classifier = FakeTextModule(
        model=classifier_model,
        lr=TRAIN_LR,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
    )

    seed_everything(RANDOM_SEED, workers=True)

    train_model(
        model=text_classifier,
        data_module=text_data_module,
        epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=logger,
    )
