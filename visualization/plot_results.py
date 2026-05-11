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

import matplotlib.pyplot as plt


def plot_data(X_test, y_test):

    plt.figure(figsize=(8, 6))

    for label in [0, 1]:

        plt.scatter(
            X_test[y_test == label][:, 0],
            X_test[y_test == label][:, 1],
            label=f"Class {label}"
        )

    plt.title("Quantum Machine Learning Classification")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.legend()
    plt.grid(True)

    plt.show()