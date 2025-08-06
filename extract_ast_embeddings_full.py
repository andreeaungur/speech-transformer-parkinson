import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

SAMPLE_RATE = 16000
ROOT_DIR = "vowels"
MODEL_NAME = "./pcgita_ast_finetuned"

feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()

data = []

for label_name in ["healthy", "parkinson"]:
    label_dir = os.path.join(ROOT_DIR, label_name)
    label = 0 if label_name == "healthy" else 1

    print(f"\n Scanning {label_name.upper()} from {label_dir}")

    for root, _, files in os.walk(label_dir):
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        for file in tqdm(wav_files, desc=f"{label_name} - {os.path.basename(root)}"):
            file_path = os.path.join(root, file)

            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                waveform = waveform.squeeze()

                inputs = feature_extractor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # ultima stare ascunsă
                    embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

                data.append({
                    "path": file_path,
                    "label": label,
                    "embedding": embedding
                })

            except Exception as e:
                print(f" Error at {file_path}: {e}")
                continue

df = pd.DataFrame(data)
df.to_pickle("ast_embeddings_vowels_full.pkl")
print(f"\n Saved: ast_embeddings_vowels_full.pkl cu {len(df)} files.")
