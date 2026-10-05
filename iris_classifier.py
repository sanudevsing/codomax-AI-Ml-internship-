"""
Iris Flower Classification - First Machine Learning Project (Scikit-learn)

The basic ML workflow:
  1. Load data
  2. Explore data
  3. Split into training and test sets
  4. Train a model
  5. Evaluate on unseen (test) data
  6. Compare models
  7. Predict on new flowers

Run with:  python iris_classifier.py
Requires:  pip install scikit-learn pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # remove this line if you want plt.show() windows
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# =====================================================
# 1. LOAD DATA
# =====================================================
iris = load_iris(as_frame=True)
X = iris.data                      # features: 4 measurements (cm)
y = iris.target                    # label: 0, 1, 2
class_names = iris.target_names    # ['setosa' 'versicolor' 'virginica']

# =====================================================
# 2. EXPLORE DATA
# =====================================================
print("--- First 5 rows ---")
print(X.head())
print(f"\nSamples: {X.shape[0]}, Features: {X.shape[1]}")
print("\nSamples per species:")
print(y.map(dict(enumerate(class_names))).value_counts())
print("\nAverage measurements per species:")
print(X.groupby(y.map(dict(enumerate(class_names)))).mean().round(2))

# Scatter plot: can we already see the groups?
plt.figure(figsize=(6, 5))
for i, name in enumerate(class_names):
    mask = y == i
    plt.scatter(X.loc[mask, "petal length (cm)"], X.loc[mask, "petal width (cm)"], label=name)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Iris species by petal size")
plt.legend()
plt.tight_layout()
plt.savefig("iris_scatter.png", dpi=120)
plt.close()

# =====================================================
# 3. SPLIT: train on 80%, hold back 20% for testing
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,    # makes the split repeatable
    stratify=y,         # keeps species balanced in both sets
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# =====================================================
# 4. TRAIN YOUR FIRST MODEL
# =====================================================
# A pipeline scales the features, then fits the model.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
model.fit(X_train, y_train)
print("\nModel trained!")

# =====================================================
# 5. EVALUATE ON THE TEST SET
# =====================================================
y_pred = model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=120)
plt.close()

# Look at the individual predictions
results = X_test.copy()
results["actual"] = [class_names[i] for i in y_test]
results["predicted"] = [class_names[i] for i in y_pred]
results["correct"] = results["actual"] == results["predicted"]
print("\nFirst 8 test predictions:")
print(results[["petal length (cm)", "petal width (cm)", "actual", "predicted", "correct"]].head(8))
wrong = results[~results["correct"]]
print(f"\nMistakes: {len(wrong)} out of {len(results)}")

# =====================================================
# 6. COMPARE MODELS (cross-validation = more reliable than one split)
# =====================================================
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
}
print("\n5-fold cross-validation (all data):")
for name, clf in candidates.items():
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  {name:<22} mean = {scores.mean():.2%}  (+/- {scores.std():.2%})")

# =====================================================
# 7. PREDICT NEW FLOWERS
# =====================================================
new_flowers = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2],     # looks like a setosa
     [6.0, 2.9, 4.5, 1.5],     # looks like a versicolor
     [6.9, 3.1, 5.8, 2.1]],    # looks like a virginica
    columns=X.columns,
)
predictions = model.predict(new_flowers)
probabilities = model.predict_proba(new_flowers)

print("\nPredictions for new flowers:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities), start=1):
    print(f"  Flower {i}: {class_names[pred]:<11} (confidence {prob.max():.1%})")

# Which measurements mattered most?
coefs = np.abs(model[-1].coef_).mean(axis=0)
print("\nFeature influence (larger = more important):")
for feature, c in sorted(zip(X.columns, coefs), key=lambda t: -t[1]):
    print(f"  {feature:<20} {c:.2f}")
