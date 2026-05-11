# =========================================================
# COPYRIGHT NOTICE
# =========================================================
#
# Project Title:
# Quantum Cancer Detection using Qiskit
#
# Author:
# Piyush Anandrao Shewale
#
# Copyright (c) 2026 Piyush Anandrao Shewale
#
# All Rights Reserved.
#
# This project was developed for academic, educational,
# and research demonstration purposes.
#
# This project demonstrates the application of:
# - Quantum Computing
# - Quantum Machine Learning
# - Variational Quantum Circuits (VQC)
# - Healthcare Data Classification
#
# Technologies Used:
# - Python
# - Qiskit
# - Qiskit Machine Learning
# - Scikit-learn
# - NumPy
# - Matplotlib
# - Seaborn
#
# Unauthorized commercial redistribution,
# reproduction, modification, or claiming this
# work as original without proper credit is discouraged.
#
# Quantum computing frameworks and libraries belong
# to their respective owners and organizations.
#
# DISCLAIMER:
# This project is intended only for educational
# and research demonstration purposes.
#
# This is NOT a real medical diagnostic system.
#
# =========================================================

from data.dataset import load_dataset
from data.preprocess import preprocess_data

from model.quantum_model import create_quantum_model
from model.train import train_model

from utils.metrics import evaluate_model

from visualization.confusion_matrix_plot import plot_confusion_matrix

import numpy as np


def main():

    # =====================================================
    # LOAD DATASET
    # =====================================================

    X, y, feature_names = load_dataset()

    print("========================================")
    print(" Breast Cancer Dataset Loaded ")
    print("========================================")

    print("\nTotal Features:", len(feature_names))
    print("Total Samples :", len(X))

    # =====================================================
    # PREPROCESS DATA
    # =====================================================

    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(X, y)

    print("\n========================================")
    print(" Dataset Preprocessing Complete ")
    print("========================================")

    print("\nTraining Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    # =====================================================
    # CREATE QUANTUM MODEL
    # =====================================================

    model = create_quantum_model()

    print("\n========================================")
    print(" Quantum Model Created ")
    print("========================================")

    # =====================================================
    # DISPLAY QUANTUM CIRCUITS
    # =====================================================

    print("\nFeature Map Circuit:\n")
    print(model.feature_map)

    print("\nTrainable Ansatz Circuit:\n")
    print(model.ansatz)

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    trained_model = train_model(
        model,
        X_train,
        y_train
    )

    # =====================================================
    # EVALUATE MODEL
    # =====================================================

    predictions, accuracy, report, matrix = evaluate_model(
        trained_model,
        X_test,
        y_test
    )

    print("\n========================================")
    print(" MODEL PERFORMANCE ")
    print("========================================")

    print("\nQuantum Model Accuracy:")
    print(round(accuracy * 100, 2), "%")

    print("\nClassification Report:\n")
    print(report)

    print("\nConfusion Matrix:\n")
    print(matrix)

    # =====================================================
    # SHOW CONFUSION MATRIX GRAPH
    # =====================================================

    plot_confusion_matrix(matrix)

    # =====================================================
    # USER INPUT SECTION
    # =====================================================

    print("\n========================================")
    print(" QUANTUM CANCER PREDICTION ")
    print("========================================")

    print("\nEnter sample medical values.\n")

    print("Typical Mean Radius Range : 6 - 30")
    print("Typical Mean Texture Range: 10 - 40\n")

    # Take user input
    radius = float(input("Enter Mean Radius : "))
    texture = float(input("Enter Mean Texture: "))

    # =====================================================
    # CREATE INPUT SAMPLE
    # =====================================================

    sample = np.array([[radius, texture]])

    # Pad remaining 28 features with zeros
    sample = np.pad(
        sample,
        ((0, 0), (0, 28)),
        mode='constant'
    )

    # =====================================================
    # APPLY SAME PREPROCESSING
    # =====================================================

    sample = scaler.transform(sample)

    sample = pca.transform(sample)

    # =====================================================
    # PREDICT USING QUANTUM MODEL
    # =====================================================

    prediction = trained_model.predict(sample)

    # Safe conversion
    prediction_value = int(prediction)

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    print("\n========================================")
    print(" PREDICTION RESULT ")
    print("========================================")

    if prediction_value == 0:

        print("\nMalignant Tumor Detected")
        print("Cancer Presence: YES")

    else:

        print("\nBenign Tumor Detected")
        print("Cancer Presence: NO")

    print("\n========================================")
    print(" Educational Use Only ")
    print(" Not for real medical diagnosis ")
    print("========================================")


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()