import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torch import optim, nn

# 1. Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. Configuration
ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz"
LABELS = {"healthy": ["hc", "HC", "Control"], "parkinson": ["pd", "PD", "Patologicas", "Patologica"]}

# 3. Label inference
def infer_label(path):
    path = path.lower()
    for key in LABELS["healthy"]:
        if key.lower() in path:
            return 0
    for key in LABELS["parkinson"]:
        if key.lower() in path:
            return 1
    return None

# 4. DDK file check - extended with more variants found in PC-GITA
def is_ddk_file(path):
    path_lower = path.lower()
    keywords = [
        "ddk analysis", "ka-ka-ka", "pa-pa-pa", "pakata",
        "pataka", "petaka", "ta-ta-ta"
    ]
    return any(k in path_lower for k in keywords)

# 5. Collect files
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            if is_ddk_file(full_path):
                label = infer_label(full_path)
                if label is not None:
                    all_files.append((full_path, label))

print(f"Total DDK files found: {len(all_files)}")

# Additional check to avoid crash
if len(all_files) == 0:
    raise RuntimeError(
        "No DDK files found! "
        " Check if ROOT_DIR path is correct "
        " and if files contain in their name one of the keywords from is_ddk_file()."
    )

# 6. Split
train_files, val_files = train_test_split(
    all_files, test_size=0.2, random_state=42, stratify=[lbl for _, lbl in all_files]
)


# 7. Dataset
class PCGitaDDKDataset(Dataset):
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
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        waveform = waveform.squeeze().numpy()
        if self.augment:
            waveform = self.augmenter(samples=waveform, sample_rate=16000)

        input_features = self.feature_extractor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_values[0]

        return {"input_values": input_features, "labels": torch.tensor(label)}

# 8. Load model + extractor
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=2,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 9. Dataloaders
train_dataset = PCGitaDDKDataset(train_files, feature_extractor, augment=True)
val_dataset = PCGitaDDKDataset(val_files, feature_extractor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 10. Optimizer & loss
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 11. Manual training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    
    # Training
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # Evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    print(f"Validation Accuracy: {acc:.4f}")

# 12. Save model & extractor
model.save_pretrained("./pcgita_ast_finetuned_ddk")
feature_extractor.save_pretrained("./pcgita_ast_finetuned_ddk")
print("Fine-tuned AST model on DDK saved in './pcgita_ast_finetuned_ddk'")
