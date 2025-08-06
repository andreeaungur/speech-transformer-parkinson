import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

# Load the embeddings
df = pd.read_pickle("ast_embeddings_vowels_full.pkl")

print(f" Total examples: {len(df)}")
print(f" Unique labels: {np.unique(df['label'])}")

# Prepare data 
X = np.stack(df["embedding"].values)
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize scalers for SVM and MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classificators
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300)
}

results = []

for name, clf in models.items():
    print(f"\n Evaluating the model: {name}")

    # Choose scaled data or not
    if name in ["SVM (Linear)", "MLP Classifier"]:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    acc = report["accuracy"]
    print(f" Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson"]))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision_Healthy": report["0"]["precision"],
        "Recall_Healthy": report["0"]["recall"],
        "F1_Healthy": report["0"]["f1-score"],
        "Precision_Parkinson": report["1"]["precision"],
        "Recall_Parkinson": report["1"]["recall"],
        "F1_Parkinson": report["1"]["f1-score"]
    })

# Save the results
results_df = pd.DataFrame(results)
results_df.to_csv("ast_classification_results.csv", index=False)
print("\n Classification is over. The results were saved in: ast_classification_results.csv")
