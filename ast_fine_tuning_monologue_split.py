import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise
from tqdm import tqdm

# -------------------------
# 1. Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------------
# 2. Configuration
ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz/monologue/sin normalizar"
LABELS = {"healthy": ["hc"], "parkinson": ["pd"]}

# -------------------------
# 3. Label inference
def infer_label(path):
    parts = path.lower().split(os.sep)
    for key in LABELS["healthy"]:
        if key in parts:
            return 0
    for key in LABELS["parkinson"]:
        if key in parts:
            return 1
    return None

# -------------------------
# 4. Speaker ID
def get_speaker_id(path):
    return os.path.basename(path)[:13]  

# -------------------------
# 5. Simple energy-based VAD
def simple_vad(audio, sample_rate, frame_length=1024, hop_length=512, energy_threshold=0.001, min_segment_length=0.5):
    audio = audio.numpy() if torch.is_tensor(audio) else audio
    energy = []
    for start in range(0, len(audio), hop_length):
        frame = audio[start:start+frame_length]
        energy.append(np.sum(frame**2))
    energy = np.array(energy)
    voiced_frames = energy > energy_threshold

    segments = []
    start_frame = None
    for i, voiced in enumerate(voiced_frames):
        if voiced and start_frame is None:
            start_frame = i
        elif not voiced and start_frame is not None:
            end_frame = i
            start_sample = start_frame * hop_length
            end_sample = end_frame * hop_length + frame_length
            duration = (end_sample - start_sample) / sample_rate
            if duration >= min_segment_length:
                segments.append(audio[start_sample:end_sample])
            start_frame = None
    if start_frame is not None:
        start_sample = start_frame * hop_length
        end_sample = len(audio)
        duration = (end_sample - start_sample) / sample_rate
        if duration >= min_segment_length:
            segments.append(audio[start_sample:end_sample])
    return segments

# -------------------------
# 6. Collect files
all_files = []
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            full_path = os.path.join(root, file)
            label = infer_label(full_path)
            if label is not None:
                all_files.append((full_path, label))

print(f"Total monologue files found: {len(all_files)}")

# -------------------------
# 7. Train/val split per speaker
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

print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")

# -------------------------
# 8. Dataset class
augmenter = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5)])

class SpeechSegmentsDataset(Dataset):
    def __init__(self, file_label_pairs, feature_extractor, augment=False):
        self.files = file_label_pairs
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)  # mono
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        segments = simple_vad(waveform, sr, energy_threshold=0.001, min_segment_length=0.5)
        if len(segments) > 0:
            segment = segments[0]
            if self.augment:
                segment = augmenter(samples=segment, sample_rate=16000)
        else:
            segment = waveform.numpy()

        inputs = self.feature_extractor(segment, sampling_rate=16000, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["labels"] = torch.tensor(label)
        return inputs

# -------------------------
# 9. Collate function
def collate_fn(batch):
    batch_out = {}
    for key in batch[0].keys():
        batch_out[key] = torch.stack([b[key] for b in batch])
    return batch_out

# -------------------------
# 10. Feature extractor & model
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=2, ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 11. Dataset instances
train_dataset = SpeechSegmentsDataset(train_files, feature_extractor, augment=True)
val_dataset = SpeechSegmentsDataset(val_files, feature_extractor, augment=False)

# -------------------------
# 12. Trainer setup
training_args = TrainingArguments(
    output_dir="./results_ast_monologue",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_dir="./logs_ast_monologue",
    logging_steps=10,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    print(f"Evaluation accuracy: {acc:.4f}")
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# -------------------------
# 13. Training loop
for epoch in range(training_args.num_train_epochs):
    print(f"\n==== Start training epoch {epoch + 1} ====")
    trainer.train(resume_from_checkpoint=False)
    print(f"\n==== Evaluate after epoch {epoch + 1} ====")
    metrics = trainer.evaluate()
    print(metrics)

# -------------------------
# 14. Save model & feature extractor
model.save_pretrained("/content/drive/MyDrive/pcgita_ast_monologue")
feature_extractor.save_pretrained("/content/drive/MyDrive/pcgita_ast_monologue")
print("Fine-tuned AST model on Monologue saved.")
