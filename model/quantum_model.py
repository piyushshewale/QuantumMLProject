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

from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.algorithms import VQC


def create_quantum_model():

    # Quantum feature encoding
    feature_map = ZZFeatureMap(
        feature_dimension=2,
        reps=2
    )

    # Trainable quantum circuit
    ansatz = RealAmplitudes(
        num_qubits=2,
        reps=2
    )

    # Variational Quantum Classifier
    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz
    )

    return model