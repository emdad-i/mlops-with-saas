import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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

# 5. Save the trained model artifact
joblib.dump(model, "app/model.pkl")
print("Model saved to app/model.pkl")

# Close the wandb run
wandb.finish()