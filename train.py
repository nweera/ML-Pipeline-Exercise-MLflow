import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name, params):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=params.get("max_iter", 200))

    elif model_name == "decision_tree":
        return DecisionTreeClassifier(max_depth=params.get("max_depth", None))

    elif model_name == "random_forest":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 100))

    else:
        raise ValueError(f"Unknown model type: {model_name}")


def main():

    # -----------------------
    # Load config file
    # -----------------------
    config = load_config()

    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_csv(config["data"]["path"])
    target_col = config["data"]["target_column"]

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        shuffle=config["data"]["shuffle"],
    )

    # -----------------------
    # Train each model
    # -----------------------
    for model_name, params in config["models"].items():
        print(f"\n🚀 Training {model_name}...")

        model = build_model(model_name, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        if config["training"]["save_metrics"]:
            output_path = config["training"]["metrics_output_pattern"].format(
                model=model_name
            )
            with open(output_path, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {acc:.4f}\n")

            print(f"✅ Saved: {output_path}")

        print(f"📊 Accuracy for {model_name}: {acc:.4f}")


if __name__ == "__main__":
    main()
