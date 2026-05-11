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

from sklearn.datasets import load_breast_cancer


def load_dataset():

    cancer = load_breast_cancer()

    X = cancer.data
    y = cancer.target

    feature_names = cancer.feature_names

    return X, y, feature_names