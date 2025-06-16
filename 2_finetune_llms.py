import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class ClinVarClassificationPipeline:
    def __init__(
        self,
        model_path,
        data_path,
        label_column,
        output_dir,
        max_len=246,
        batch_size=8,
        epochs=10,
        lr=2e-5,
        patience=1,
        multiclass=False,
        dropna_subset=["Sequence"],
        label_dtype=int,
        stratified_split=True,
        weighted_loss=True,
        device=None,
    ):
        self.model_path   = model_path
        self.data_path    = data_path
        self.label_column = label_column
        self.output_dir   = output_dir
        self.max_len      = max_len
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.lr           = lr
        self.patience     = patience
        self.multiclass   = multiclass
        self.stratified_split = stratified_split
        self.weighted_loss = weighted_loss
        self.dropna_subset = dropna_subset + [label_column]
        self.label_dtype  = label_dtype
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)
        self.tokenizer    = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)

        # print details of which model is running
        print("="*60)
        print(f"Starting fine-tuning pipeline...")
        print(f"  Model path   : {self.model_path}")
        print(f"  Dataset path : {self.data_path}")
        print(f"  Output dir   : {self.output_dir}")
        print(f"  Label column : {self.label_column}")
        print(f"  Max sequence length: {self.max_len}")
        print(f"  Batch size   : {self.batch_size}")
        print(f"  Epochs       : {self.epochs}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Multiclass   : {self.multiclass}")
        print("="*60)
        
        self._prepare_dataset()
        self._prepare_model()
        self._prepare_loaders()

    def _prepare_dataset(self):
        df = pd.read_csv(self.data_path, sep="\t", dtype=str)
        df = df.dropna(subset=self.dropna_subset)
        if not self.multiclass:
            df[self.label_column] = df[self.label_column].astype(self.label_dtype)
        else:
            df[self.label_column] = df[self.label_column].astype(int)
        self.df = df
        self.num_labels = df[self.label_column].nunique()
        self.labels = df[self.label_column].values

        class CustomDataset(Dataset):
            def __init__(self, dataframe, tokenizer, label_column, max_length):
                self.sequences = dataframe["Sequence"].tolist()
                self.labels    = dataframe[label_column].tolist()
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                seq   = self.sequences[idx]
                label = self.labels[idx]
                enc   = self.tokenizer(
                    seq,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                item = {k: v.squeeze(0) for k, v in enc.items()}
                item["labels"] = torch.tensor(label, dtype=torch.long)
                return item

        self.dataset = CustomDataset(self.df, self.tokenizer, self.label_column, self.max_len)

    def _prepare_model(self):
        config = AutoConfig.from_pretrained(self.model_path, num_labels=self.num_labels, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
        emb_shape = self.model.base_model.embeddings.word_embeddings.weight.shape
        vocab_sz  = self.tokenizer.vocab_size
        if emb_shape[0] != vocab_sz:
            raise ValueError(
                f"Model embedding shape {emb_shape} does not match tokenizer vocab size {vocab_sz}."
            )

    def _prepare_loaders(self):
        from sklearn.model_selection import train_test_split

        if self.stratified_split:
            train_idx, val_idx = train_test_split(
                np.arange(len(self.df)),
                test_size=0.2,
                stratify=self.labels,
                random_state=42,
            )
            train_ds = torch.utils.data.Subset(self.dataset, train_idx)
            val_ds   = torch.utils.data.Subset(self.dataset, val_idx)
        else:
            train_size = int(0.8 * len(self.dataset))
            val_size   = len(self.dataset) - train_size
            train_ds, val_ds = random_split(
                self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(val_ds, batch_size=self.batch_size)
        if self.stratified_split:
            self.y_train = self.labels[train_idx]
        else:
            self.y_train = self.labels[:len(self.train_loader.dataset)]

    def _get_class_weights(self):
        if self.weighted_loss:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(self.y_train)
            weights = compute_class_weight("balanced", classes=classes, y=self.y_train)
            print("Using class weights:", dict(zip(classes, weights)))
            return torch.tensor(weights, dtype=torch.float).to(self.device)
        else:
            return None

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        total_steps = self.epochs * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        class_weights = self._get_class_weights()

        if self.multiclass:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else torch.nn.CrossEntropyLoss()
        else:
            if self.num_labels == 2:
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else torch.nn.CrossEntropyLoss()
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights) if class_weights is not None else torch.nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        no_improve_epochs = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_epoch = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits  = outputs.logits
                labels  = batch["labels"]
                if self.multiclass or self.num_labels > 2:
                    loss = loss_fn(logits, labels)
                else:
                    loss = loss_fn(logits, labels)
                total_train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    logits  = outputs.logits
                    labels  = batch["labels"]
                    if self.multiclass or self.num_labels > 2:
                        loss = loss_fn(logits, labels)
                        preds = torch.argmax(logits, dim=-1)
                    else:
                        loss = loss_fn(logits, labels)
                        preds = torch.argmax(logits, dim=-1)
                    total_val_loss += loss.item()
                    correct += (preds == labels).sum().item()
                    total += preds.size(0)
            avg_val_loss = total_val_loss / len(self.val_loader)
            val_acc = correct / total if total > 0 else 0
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            print(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                best_epoch = epoch
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                print(f"Validation loss improved. Model saved to {self.output_dir}.")
            else:
                no_improve_epochs += 1
                print(f"No improvement for {no_improve_epochs} epoch(s).")
                if no_improve_epochs >= self.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best epoch was {best_epoch}."
                    )
                    break

        # Plot Loss and Accuracy
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses,   label="Val Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))

        plt.figure(figsize=(8, 4))
        plt.plot(val_accuracies, label="Val Accuracy")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.savefig(os.path.join(self.output_dir, "accuracy_curve.png"))

        print("Training complete. Curves saved in the output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Sequence Classification Finetuning")
    parser.add_argument("--model_path",    type=str, required=True, help="Path to pretrained model directory")
    parser.add_argument("--data_path",     type=str, required=True, help="Path to input .tsv file")
    parser.add_argument("--label_column",  type=str, required=True, help="Column name for target labels")
    parser.add_argument("--output_dir",    type=str, required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--max_len",       type=int, default=246, help="Max sequence length")
    parser.add_argument("--batch_size",    type=int, default=8, help="Batch size")
    parser.add_argument("--epochs",        type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr",            type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--patience",      type=int, default=1, help="Patience for early stopping")
    parser.add_argument("--multiclass",    action="store_true", help="Set if multi-class classification")
    parser.add_argument("--label_dtype",   type=str, default="int", help="Data type for labels (int, float, etc.)")
    parser.add_argument("--stratified_split", action="store_true", help="Use stratified split (default: True)")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss for class imbalance")
    args = parser.parse_args()

    dtype_map = {"int": int, "float": float, "str": str}
    label_dtype = dtype_map.get(args.label_dtype, int)

    pipeline = ClinVarClassificationPipeline(
        model_path   = args.model_path,
        data_path    = args.data_path,
        label_column = args.label_column,
        output_dir   = args.output_dir,
        max_len      = args.max_len,
        batch_size   = args.batch_size,
        epochs       = args.epochs,
        lr           = args.lr,
        patience     = args.patience,
        multiclass   = args.multiclass,
        label_dtype  = label_dtype,
        stratified_split=args.stratified_split,
        weighted_loss=args.weighted_loss,
    )
    pipeline.train()


