import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Generate Large Synthetic Dataset
# ----------------------------

np.random.seed(42)

n = 500

income = np.random.randint(20000, 150000, n)
age = np.random.randint(21, 60, n)
loan_amount = np.random.randint(5000, 50000, n)
credit_history = np.random.randint(0, 2, n)

# Target logic (simple rule-based)
target = (income > 50000) & (credit_history == 1)
target = target.astype(int)

df = pd.DataFrame({
    "income": income,
    "age": age,
    "loan_amount": loan_amount,
    "credit_history": credit_history,
    "target": target
})

print("\nDataset Shape:", df.shape)

# ----------------------------
# Feature & Target Split
# ----------------------------

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", auc_score)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()