from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, max_depth=6):
    model = RandomForestClassifier(
        max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate classification model and return metrics.
    """
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds) * 100
    precision = precision_score(y_test, preds, zero_division=0) * 100
    recall = recall_score(y_test, preds, zero_division=0) * 100
    f1 = f1_score(y_test, preds, zero_division=0) * 100

    print(f"\n{model_name} Performance:")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"Precision : {precision:.2f}%")
    print(f"Recall    : {recall:.2f}%")
    print(f"F1 Score  : {f1:.2f}%")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }