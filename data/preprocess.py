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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def preprocess_data(X, y):

    # Normalize data
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # Reduce dimensions for quantum compatibility
    pca = PCA(n_components=2)

    X = pca.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, pca