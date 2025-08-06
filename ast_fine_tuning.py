import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification #, TrainingArguments, Trainer
from transformers.training_args import TrainingArguments
from transformers import Trainer

from sklearn.model_selection import train_test_split


# 1. Initial settings
ROOT_DIR = "C:/Users/Andreea/upm_project/PC-GITA_16kHz/PC-GITA_16kHz"
LABELS = {"healthy": ["hc", "HC", "Control"], "parkinson": ["pd", "PD", "Patologicas", "Patologica"]}

# 2. Labels
def infer_label(path):
    path = path.lower()
    for key in LABELS["healthy"]:
        if key.lower() in path:
            return 0
    for key in LABELS["parkinson"]:
        if key.lower() in path:
            return 1
    return None

# 3. Extracting audio file paths and corresponding labels
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            label = infer_label(full_path)
            if label is not None:
                all_files.append((full_path, label))

# 4. Split train/test
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42, stratify=[lbl for _, lbl in all_files])

# 5. Dataset custom
class PCGitaASTDataset(Dataset):
    def __init__(self, file_label_pairs, feature_extractor, max_length=1024):
        self.data = file_label_pairs
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=1024,
            hop_length=320,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

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

        waveform = waveform.squeeze().numpy()  # convert to numpy array

        input_features = self.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        return {"input_values": input_features, "labels": torch.tensor(label)}


# 6. Load AST and feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=2,
    ignore_mismatched_sizes=True 
)

# 7. Datasets
train_dataset = PCGitaASTDataset(train_files, feature_extractor)
val_dataset = PCGitaASTDataset(val_files, feature_extractor)

# 8. Args for Trainer
training_args = TrainingArguments(
    output_dir="./results_ast",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
)

# 9. Trainer 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10. Start fine-tuning 
trainer.train()

# 11. Save the trained model 
model.save_pretrained("./pcgita_ast_finetuned")
feature_extractor.save_pretrained("./pcgita_ast_finetuned")
