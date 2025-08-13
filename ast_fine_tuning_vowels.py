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

# 1. Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Config
ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz"
LABELS = {"healthy": ["hc", "HC", "Control"], "parkinson": ["pd", "PD", "Patologicas", "Patologica"]}
TARGET_SUBFOLDER = "vowels"  # only files in folders containing this word are considered
SAMPLE_RATE = 16000

# 3. Label inference function
def infer_label(path):
    path = path.lower()
    for key in LABELS["healthy"]:
        if key.lower() in path:
            return 0
    for key in LABELS["parkinson"]:
        if key.lower() in path:
            return 1
    return None

# 4. Select WAV files only from subfolders containing "vowels"
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    if TARGET_SUBFOLDER in root.lower():
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                label = infer_label(full_path)
                if label is not None:
                    all_files.append((full_path, label))

print(f"Total WAV files in '{TARGET_SUBFOLDER}': {len(all_files)}")

# 5. Train/test split
train_files, val_files = train_test_split(
    all_files, test_size=0.2, random_state=42, stratify=[lbl for _, lbl in all_files]
)

# 6. Custom Dataset cu augmentare
class PCGitaVowelDataset(Dataset):
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
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        waveform = waveform.squeeze().numpy()

        # Augment only if requested
        if self.augment:
            waveform = self.augmenter(samples=waveform, sample_rate=SAMPLE_RATE)

        inputs = self.feature_extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_values[0]

        return {"input_values": inputs, "labels": torch.tensor(label)}

# 7. Collate function for padding batches
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    padded = feature_extractor.pad({"input_values": input_values}, return_tensors="pt")
    padded["labels"] = labels
    return padded

# 8. Load pre-trained AST model and feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=2,
    ignore_mismatched_sizes=True
)

# 9. Create datasets 
train_dataset = PCGitaVowelDataset(train_files, feature_extractor, augment=True)
val_dataset = PCGitaVowelDataset(val_files, feature_extractor, augment=False)

# 10. Training arguments 
training_args = TrainingArguments(
    output_dir="./results_ast_vowels",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs_vowels",
    save_total_limit=1,
)

# 11. Metrics calculation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 12. Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# 13. Manual training loop with evaluation after each epoch
num_epochs = training_args.num_train_epochs
for epoch in range(num_epochs):
    print(f"=== Epoch {epoch + 1}/{num_epochs} ===")
    trainer.train()
    print("Evaluare după epoca", epoch + 1)
    metrics = trainer.evaluate()
    print(metrics)
    trainer.save_model(training_args.output_dir)  # Saves the model after each epoch

# 14. Save the fine-tuned model and feature extractor 
model.save_pretrained("./pcgita_ast_finetuned_vowels")
feature_extractor.save_pretrained("./pcgita_ast_finetuned_vowels")

print("Fine-tuned AST model on vowels saved in './pcgita_ast_finetuned_vowels'")
