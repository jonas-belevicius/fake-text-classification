"""
This module defines the PyTorch modules for the fake news classification task.
The `FakeTextDataset` class is a custom dataset class that loads the text data
and labels for the fake news classification task.

The `FakeTextDataModule` class is a PyTorch Lightning data module that handles
data preprocessing, tokenization, and dataloader creation for the fake news
classification task.

The `FakeTextModule` class is a PyTorch Lightning module that defines the
training and validation steps for the fake news classification task.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
)
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerBase
from typing import (
    Optional,
    List,
    Dict,
)
from config import (
    NUM_WORKERS,
    DEVICE,
    BATCH_SIZE,
    MAX_SEQ_LEN,
)


class FakeTextDataset(Dataset):
    """
    A custome dataset for loading text data with labels for text status:
    Fake / True.

    Args:
    data (pd.DataFrame): The dataframe containing text and labels.
    tokenizer (PreTrainedTokenizerBase): Tokenizer to encode the texts.
    max_seq_len (int, optional): maximum sequence length for tokenization.
    Defaults to MAX_SEQ_LEN

    Methods:
    __len__() -> int
        Returns the total number of texts in the dataset.
    __getitem__(idx: int) -> Tuple[Any, int, int]
        Returns the processed text and its corresponding label.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()

        assert isinstance(data, pd.DataFrame), '"data" must be pd.DataFrame'
        assert isinstance(max_seq_len, int), '"max_seq_len" must be an int'

        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.encoder = LabelEncoder()

        self.data["status"] = self.data["status"].astype(str)
        self.data["encoded_labels"] = self.encoder.fit_transform(self.data["status"])

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, batch_idx: int) -> int:
        """Fetches data sample by index.

        Args:
            batch_idx (int): index of the sample.

        Return:
            dict: encoded text and corresponding labels.
        """
        if torch.is_tensor(batch_idx):
            batch_idx = batch_idx.tolist()

        batch_row = self.data.iloc[batch_idx]
        text = batch_row["text"]
        label = batch_row["encoded_labels"]

        text_endoded = self.tokenizer.encode_plus(
            text=text,
            max_length=self.max_seq_len,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        text_encoded = {k: v.squeeze() for k, v in text_endoded.items()}
        encoded_dict = {
            "text": text,
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return encoded_dict


class FakeTextDataModule(pl.LightningDataModule):
    """
    A Pytorch Lightning data module for the fake/true article dataset.
    Hndles data preprocessing, tokenization and dataloader creation.

    Args:
    train_data (pd.DataFrame): Data containing text and labels for train set.
    val_data (pd.DataFrame): Data containing text and labels for validation set.
    test_data (pd.DataFrame): Data containing text and labels for test set.
    tokenizer (PreTrainedTokenizerBase): Tokenizer to encode the texts.
    batch_size (int, optional): Batch size for the dataloader.
    Defaults to BATCH_SIZE.
    max_seq_len (int, optional): Maximum sequence length for tokenization.
    Defaults to MAX_SEQ_LEN.
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = BATCH_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def clean_nan_values(self, df, text_column="text"):
        """
        Replaces NaN values in the specified text column with empty
        strings.
        """
        df[text_column] = df[text_column].fillna("")
        return df

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Splits the data into train, validation and test sets.
        Cleans NaN values in the text column.
        Tokenizes the text data and creates the corresponding datasets.
        """
        self.train_data = self.clean_nan_values(self.train_data)
        self.val_data = self.clean_nan_values(self.val_data)
        self.test_data = self.clean_nan_values(self.test_data)

        self.train_dataset = FakeTextDataset(
            self.train_data, self.tokenizer, self.max_seq_len
        )
        self.val_dataset = FakeTextDataset(
            self.val_data, self.tokenizer, self.max_seq_len
        )
        self.test_dataset = FakeTextDataset(
            self.test_data, self.tokenizer, self.max_seq_len
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device = DEVICE,
        dataloader_idx: Optional[int] = None,
    ) -> dict:
        """Transfers a batch of data to the appropriate device (CPU/GPU)."""
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        batch["labels"] = batch["labels"].to(device)

        return batch


class FakeTextModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weights: Optional[List[float]] = None,
        dropout: float = 0.3,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Initializes the FakeTextModule.

        Attributes:
        model (nn.Module):
            The backbone model for feature extraction.
        lr (float, optional):
            Learning rate for the optimizer. Defaults to 1e-3.
        weights (Optional[List[float]], optional):
            Class weights for the classification loss.
            Defaults to None (no weighting).
        dropout (float, optional):
            Dropout rate for the task-specific heads. Defaults to 0.3.
        weight_decay (float, optional):
            Weight decay for the optimizer. Defaults to 1e-4.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.weights = weights
        self.dropout = dropout
        self.epoch_start_time: Optional[float] = None
        self.cuda = DEVICE
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.train_logits = torch.empty(0, device=self.cuda)
        self.train_labels = torch.empty(
            0,
            dtype=torch.long,
            device=self.cuda,
        )

        self.val_logits = torch.empty(0, device=self.cuda)
        self.val_labels = torch.empty(
            0,
            dtype=torch.long,
            device=self.cuda,
        )

        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary", num_classes=2, average="macro")

        num_features = self.model.classifier.in_features
        self.model_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2),
        )
        self.model.classifier = self.model_head

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def model_grad(self, requires_grad: bool = False) -> None:
        """
        Sets the gradient computation for the model.

        Args:
        requires_grad (bool, optional): If False, disables gradient computation
        for the model. Default is False.
        """

        for param in list(self.model.parameters()):
            param.requires_grad = requires_grad

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = self.model(input_ids, attention_mask)
        return output

    def step(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """
        Generic step function for training and validation steps.
        Args:
            batch: A batch of data.
            - x (torch.Tensor): Input image tensors.
            - age_labels (torch.Tensor): Age labels.
            - gender_labels (torch.Tensor): Gender labels.
            mode: "train" or "val" indicating the mode of the step.
        Returns:
            loss (torch.Tensor): The total computed loss for the batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        output = self(input_ids, attention_mask)
        loss = self.criterion(output.logits, labels)

        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            f"{mode}_accuracy",
            self.accuracy(
                torch.argmax(output.logits, dim=1),
                labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{mode}_precision",
            self.precision(
                torch.argmax(output.logits, dim=1),
                labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{mode}_recall",
            self.recall(
                torch.argmax(output.logits, dim=1),
                labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{mode}_f1",
            self.f1(
                torch.argmax(output.logits, dim=1),
                labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        if mode == "train":
            self.train_logits = (
                torch.cat([self.train_logits, output.logits], dim=0)
                if self.train_logits is not None
                else output.logits
            )

            self.train_labels = (
                torch.cat([self.train_labels, labels], dim=0)
                if self.train_labels is not None
                else labels
            )

        elif mode == "val":
            self.val_logits = (
                torch.cat([self.val_logits, output.logits], dim=0)
                if self.val_logits is not None
                else output.logits
            )

            self.val_labels = (
                torch.cat([self.val_labels, labels], dim=0)
                if self.val_labels is not None
                else labels
            )

        return loss

    def on_train_epoch_start(self) -> None:
        """
        Logs the epoch start time for tracking epoch duration.
        """
        self.epoch_start_time = time.time()
        print("Epoch start time:", self.epoch_start_time)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:

        for name, param in self.named_parameters():
            if param.grad is not None:
                self.log(
                    f"grad_norm_{name}",
                    param.grad.norm(),
                    on_step=True,
                    logger=True,
                )
        return self.step(batch, mode="train")

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of the training epoch to compute and log additional
        metrics.
        Computes the AUROC for the classification task over the entire
        epoch. Resets the accumulated logits and labels for the next epoch.
        """
        auroc = self.auroc(
            torch.softmax(self.train_logits, dim=1)[:, 1],
            self.train_labels,
        )
        self.log(
            "train_auroc",
            auroc,
            on_epoch=True,
            logger=True,
        )
        self.train_logits = self.train_logits.new_empty(0)
        self.train_labels = self.train_labels.new_empty(0, dtype=torch.long)

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:

        return self.step(batch, mode="val")

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch to compute and log additional
        metrics.
        Computes the AUROC for the classification task over the entire
        epoch. Logs the duration of the epoch and resets the accumulated
        logits and labels for the next epoch.
        """
        auroc = self.auroc(
            torch.softmax(self.val_logits, dim=1)[:, 1],
            self.val_labels,
        )
        self.log(
            "val_auroc",
            auroc,
            on_epoch=True,
            logger=True,
        )

        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.log(
                "epoch_duration",
                epoch_duration / 60,
                on_epoch=True,
                logger=True,
            )
        else:
            print("Skipping epoch duration calculation for initial validation")

        self.val_logits = self.val_logits.new_empty(0)
        self.val_labels = self.val_labels.new_empty(
            0,
            dtype=torch.long,
        )

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        return [optimizer]
