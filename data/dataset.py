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

from sklearn.datasets import load_breast_cancer


def load_dataset():

    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    feature_names = cancer.feature_names

    return X, y, feature_names