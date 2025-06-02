from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging with random features and shallow trees
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5,
    max_samples=0.5,
    max_features=0.5,
    random_state=42
)

# Train the bagging classifier
bag.fit(X_train, y_train)

# Test sample
sample = X_test[0].reshape(1, -1)
print("True label:", y_test[0])
print("Individual Estimator Predictions:")
for i, est in enumerate(bag.estimators_):
    feats = bag.estimators_features_[i]
    print(f"Model {i+1}: {est.predict(sample[:, feats])[0]}")

# Bagging prediction for the sample
print("Bagging Final Prediction:", bag.predict(sample)[0])

# ------------------ Accuracy Report ------------------

# Predict on the test set
y_pred = bag.predict(X_test)

# Overall accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

# Detailed classification report
report = classification_report(y_test, y_pred, target_names=load_iris().target_names)
print("\nClassification Report:\n", report)
