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
        reps=3
    )

    # Variational Quantum Classifier
    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz
    )

    return model