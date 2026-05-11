# =========================================================
# COPYRIGHT NOTICE
# =========================================================
#
# Quantum Machine Learning Project using Qiskit
#
# Copyright (c) 2026 Piyush Shewale
#
# This project was created for academic and educational
# purposes as part of a Physics / Engineering course project.
#
# All source code, project structure, documentation,
# visualizations, and implementation logic are the
# intellectual work of the author unless otherwise stated.
#
# Technologies and libraries used:
# - Python
# - Qiskit
# - Qiskit Machine Learning
# - Scikit-learn
# - Matplotlib
#
# This project may be used for:
# - Learning
# - Academic demonstrations
# - Personal educational reference
#
# Unauthorized commercial redistribution, copying,
# or claiming this work as original without proper
# credit is discouraged.
#
# Quantum computing concepts and frameworks belong
# to their respective owners and organizations.
#
# Author:
# Piyush Anandrao Shewale
#
# Year:
# 2026
#
# =========================================================

from data.dataset import load_dataset

from model.quantum_model import create_quantum_model
from model.train import train_model

from utils.metrics import evaluate_model

from visualization.plot_results import plot_data


def main():

    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()

    print("Dataset Loaded Successfully!")

    # Create quantum model
    model = create_quantum_model()

    print("\nQuantum Circuit Created!")

    # Train model
    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate model
    predictions, accuracy = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    # Show predictions
    print("\nSample Predictions:\n")

    for i in range(5):

        print(
            f"Predicted: {predictions[i]} | Actual: {y_test[i]}"
        )

    # Plot graph
    plot_data(X_test, y_test)


if __name__ == "__main__":
    main()