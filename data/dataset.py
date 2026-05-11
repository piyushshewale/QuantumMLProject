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

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():

    # Generate dataset
    X, y = make_moons(
        n_samples=120,
        noise=0.15,
        random_state=42
    )

    # Normalize dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test