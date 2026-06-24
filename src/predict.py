import joblib
import os

models = dict()


def load_models():
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                path = os.path.join(root, file)
                models[name] = joblib.load(path)


def main():
    load_models()

    email = [input("Enter email for prediction:\n")]

    for name, pipeline in models.items():
        print(name + ":")

        prediction = pipeline.predict(email)[0]
        print("Prediction:", prediction)

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(email)[0]

            for label, prob in zip(pipeline.classes_, probabilities):
                print(f"{label}: {prob:.2%}")
        else:
            score = pipeline.decision_function(email)[0]
            print(f"Decision score: {score:.4f}")
            print("Positive score means:", pipeline.classes_[1])
            print("Negative score means:", pipeline.classes_[0])

        print()


if __name__ == "__main__":
    main()