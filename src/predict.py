import os

import joblib


def load_models() -> dict:
    loaded_models = {}

    for root, _, files in os.walk("models"):
        for file in files:
            if file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                path = os.path.join(root, file)
                loaded_models[name] = joblib.load(path)

    return loaded_models


def main() -> None:
    models = load_models()

    if not models:
        print("No trained models found. Run `python src/train.py` first.")
        return

    email = [input("Enter email for prediction:\n")]

    for name, pipeline in models.items():
        print(f"\n{name}:")

        prediction = pipeline.predict(email)[0]
        print("Prediction:", prediction)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(email)[0]

            for label, probability in zip(pipeline.classes_, probabilities):
                print(f"{label}: {probability:.2%}")
        else:
            score = pipeline.decision_function(email)[0]
            print(f"Decision score: {score:.4f}")
            print("Positive score means:", pipeline.classes_[1])
            print("Negative score means:", pipeline.classes_[0])


if __name__ == "__main__":
    main()
