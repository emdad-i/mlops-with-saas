import os
import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import wandb

# 1. Initialize wandb
wandb.init(
    project="mlops-saas-lab",
    config={
        "architecture": "RandomForest",
        "dataset": "Wine Recognition",
        "n_estimators": 100,
        "max_depth": None,
    },
)

# 2. Load benchmark data (Wine recognition dataset)
data = load_wine()
X, y = data.data, data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the model
# We use configuration parameters from wandb so they are tracked properly
model = RandomForestClassifier(
    n_estimators=wandb.config.n_estimators, max_depth=wandb.config.max_depth
)
model.fit(X_train, y_train)

# 4. Evaluate using multiple benchmark metrics
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="macro")
recall = recall_score(y_test, predictions, average="macro")
f1 = f1_score(y_test, predictions, average="macro")

# Log all benchmark metrics to wandb
wandb.log(
    {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
)

print(f"Model trained successfully. Test Accuracy: {accuracy:.4f}")

# 5. Save the trained model artifact locally
model_path = os.path.join("app", "model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 6. Upload the trained model artifact to Hugging Face Hub
hf_repo_id = os.environ.get("HUGGINGFACE_REPO_ID")
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

if hf_repo_id:
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="model.pkl",
            repo_id=hf_repo_id,
            repo_type="model",
            token=hf_token,
        )
        print(f"Model uploaded to Hugging Face repo: {hf_repo_id}/model.pkl")
    except Exception as exc:
        print("Failed to upload model to Hugging Face:", exc)
        raise
else:
    print(
        "Skipping Hugging Face upload because HUGGINGFACE_REPO_ID is not set. "
        "Set the environment variable to enable upload."
    )

# Close the wandb run
wandb.finish()