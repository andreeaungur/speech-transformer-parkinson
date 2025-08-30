import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from torch import optim, nn
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz/Vowels"
LABELS = {"healthy": ["Control"], "parkinson": ["Patologicas"]}

def infer_label(path):
    path = path.lower()
    for key in LABELS["healthy"]:
        if key.lower() in path:
            return 0
    for key in LABELS["parkinson"]:
        if key.lower() in path:
            return 1
    return None

def get_speaker_id(path):
    filename = os.path.basename(path)
    return filename[:13]
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".wav"):
            label = infer_label(os.path.dirname(root))
            if label is not None:
                all_files.append((os.path.join(root, file), label))

# Split per speaker
files_by_label = {0: [], 1: []}
for path, label in all_files:
    files_by_label[label].append((path, label))

train_files, val_files = [], []
for label, files in files_by_label.items():
    speakers = list(set(get_speaker_id(f[0]) for f in files))
    train_speakers, val_speakers = train_test_split(speakers, test_size=0.2, random_state=42)
    for f in files:
        speaker = get_speaker_id(f[0])
        if speaker in train_speakers:
            train_files.append(f)
        else:
            val_files.append(f)

class VowelsDataset(Dataset):
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
        input_values = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_values[0]
        return {"input_values": input_values, "labels": torch.tensor(label)}

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=2, ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = VowelsDataset(train_files, feature_extractor, augment=True)
val_dataset = VowelsDataset(val_files, feature_extractor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    print(f"Validation Accuracy: {acc:.4f}")

model.save_pretrained("/content/drive/MyDrive/pcgita_ast_vowels")
feature_extractor.save_pretrained("/content/drive/MyDrive/pcgita_ast_vowels")
print("Vowels model saved.")
