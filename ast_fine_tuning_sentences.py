import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# 1. Seed for reproductibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Initial settings
ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz"
LABELS = {"healthy": ["hc", "HC", "Control"], "parkinson": ["pd", "PD", "Patologicas", "Patologica"]}
VALID_FOLDERS = ["sentences", "sentences2"]  # doar aceste foldere vor fi luate în considerare

# 3. Labels
def infer_label(path):
    path = path.lower()
    for key in LABELS["healthy"]:
        if key.lower() in path:
            return 0
    for key in LABELS["parkinson"]:
        if key.lower() in path:
            return 1
    return None

# 4. Extract audio file paths and labels (only sentences și sentences2)
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    if not any(folder in root.lower() for folder in VALID_FOLDERS):
        continue
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            label = infer_label(full_path)
            if label is not None:
                all_files.append((full_path, label))

print(f"Total files from sentences/sentences2: {len(all_files)}")

# 5. Split train/test
train_files, val_files = train_test_split(
    all_files,
    test_size=0.2,
    random_state=42,
    stratify=[lbl for _, lbl in all_files]
)

# 6. Dataset custom with augmentation
class PCGitaASTDataset(Dataset):
    def __init__(self, file_label_pairs, feature_extractor, augment=False):
        self.data = file_label_pairs
        self.feature_extractor = feature_extractor
        self.augment = augment
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze().numpy()

        if self.augment:
            waveform = self.augmenter(samples=waveform, sample_rate=16000)

        input_features = self.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        return {"input_values": input_features, "labels": torch.tensor(label)}

# 7. Collate function for padding batch
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    padded = feature_extractor.pad({"input_values": input_values}, return_tensors="pt")
    padded["labels"] = labels
    return padded

# 8. Load AST and feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 9. Crate datasets (augumentation for the training part)
train_dataset = PCGitaASTDataset(train_files, feature_extractor, augment=True)
val_dataset = PCGitaASTDataset(val_files, feature_extractor, augment=False)

# 10. TrainingArguments 
training_args = TrainingArguments(
    output_dir="./results_ast_sentences",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # o epocă pe apel
    learning_rate=2e-5,
    logging_dir="./logs_ast_sentences",
)

# 11. Metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 12. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 13. Loop (manual) for fine-tuning, evaluating after each epoch
num_epochs = 3
for epoch in range(num_epochs):
    print(f"=== Epoch {epoch + 1}/{num_epochs} ===")
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Results for evaluation epoch {epoch + 1}: {eval_results}")
    trainer.save_model(f"./pcgita_ast_finetuned_sentences_model_epoch_{epoch + 1}")

# 14. Save model and final feature extractor
model.save_pretrained("./pcgita_ast_finetuned_sentences_model_final")
feature_extractor.save_pretrained("./pcgita_ast_finetuned_sentences_model_final")
