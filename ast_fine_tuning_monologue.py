import os
os.environ["WANDB_DISABLED"] = "true"
import torchaudio
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, ASTForAudioClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import random
from audiomentations import AddGaussianNoise, Compose

# 1. Simple energy-based VAD
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

# 2. Augmentation pipeline: Gaussian noise only
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
])

# 3. Dataset class for audio segments with optional augmentation
class SpeechSegmentsDataset(Dataset):
    def __init__(self, audio_segments, labels, feature_extractor, augment=False):
        self.audio_segments = audio_segments
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        audio = self.audio_segments[idx]
        label = self.labels[idx]

        # Augment only training set, 50% chance
        if self.augment and random.random() < 0.5:
            audio = augmenter(samples=audio, sample_rate=16000)

        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["labels"] = torch.tensor(label)
        return inputs

# 4. Collate function for batching
def collate_fn(batch):
    batch_out = {}
    for key in batch[0].keys():
        batch_out[key] = torch.stack([b[key] for b in batch])
    return batch_out

def main():
    ROOT_DIR = "/content/drive/MyDrive/PC-GITA_16kHz/PC-GITA_16kHz"
    LABELS = {"healthy": ["hc", "HC", "Control"], "parkinson": ["pd", "PD", "Patologicas", "Patologica"]}

    # 5. Infer label from path
    def infer_label(path):
        path = path.lower()
        for key in LABELS["healthy"]:
            if key.lower() in path:
                return 0
        for key in LABELS["parkinson"]:
            if key.lower() in path:
                return 1
        return None

    # 6. Filter monologue files
    def is_monologue_file(path):
        return "monologue" in path.lower()

    all_files = []
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                if is_monologue_file(full_path):
                    label = infer_label(full_path)
                    if label is not None:
                        all_files.append((full_path, label))

    print(f"Total monologue files found: {len(all_files)}")

    audio_segments = []
    labels = []

    feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=2, ignore_mismatched_sizes=True
    )

    for filepath, label in all_files:
        waveform, sr = torchaudio.load(filepath)
        waveform = waveform.mean(dim=0)  # mono
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000

        segments = simple_vad(waveform, sr, energy_threshold=0.001, min_segment_length=0.5)
        audio_segments.extend(segments)
        labels.extend([label] * len(segments))

    print(f"Total segments after VAD: {len(audio_segments)}")
    if len(audio_segments) == 0:
        print("Warning: No vocal segments extracted after VAD!")

    print(f"Labels distribution: {np.bincount(labels)}")

    # Train-test split stratified
    train_audio, val_audio, train_labels, val_labels = train_test_split(
        audio_segments, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Dataset instances with augmentation only for train
    train_dataset = SpeechSegmentsDataset(train_audio, train_labels, feature_extractor, augment=True)
    val_dataset = SpeechSegmentsDataset(val_audio, val_labels, feature_extractor, augment=False)

    # Training arguments - *without* evaluation_strategy
    training_args = TrainingArguments(
        output_dir="./results_ast_monologue",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="epoch",
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_dir="./logs_ast_monologue",
        logging_steps=10,
        save_total_limit=2,
    )

    # Metrics function
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

    # Training with manual evaluation after each epoch
    for epoch in range(training_args.num_train_epochs):
        print(f"\n==== Start training epoch {epoch + 1} ====")
        trainer.train(resume_from_checkpoint=False)
        print(f"\n==== Evaluate after epoch {epoch + 1} ====")
        metrics = trainer.evaluate()
        print(metrics)

    # Save model + feature extractor
    model.save_pretrained("./pcgita_ast_finetuned_monologue")
    feature_extractor.save_pretrained("./pcgita_ast_finetuned_monologue")

if __name__ == "__main__":
    main()
