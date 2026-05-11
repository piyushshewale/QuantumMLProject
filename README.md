# Quantum Cancer Detection using Qiskit

## Overview

This project demonstrates a Quantum Machine Learning (QML) based cancer detection system using Variational Quantum Circuits (VQC) built with Qiskit.

The system uses quantum computing principles to classify breast tumors as:

- Malignant (Cancer Detected)
- Benign (No Cancer Detected)

The project combines concepts of:
- Quantum Computing
- Artificial Intelligence
- Machine Learning
- Medical Data Analysis
- Quantum Classification

---

# Project Objectives

- Demonstrate Quantum Machine Learning concepts
- Apply quantum circuits to healthcare datasets
- Perform binary cancer classification
- Visualize model performance
- Build an interactive prediction system
- Understand practical applications of quantum computing

---

# Technologies Used

- Python
- Qiskit
- Qiskit Machine Learning
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

---

# Quantum Concepts Used

## Qubits

Quantum bits capable of existing in multiple states simultaneously.

## Superposition

Allows qubits to represent both 0 and 1 at the same time.

## Entanglement

Creates relationships between qubits for advanced computation.

## Quantum Gates

Operations that manipulate quantum states.

## Variational Quantum Circuits (VQC)

Parameterized trainable quantum circuits used for machine learning.

## Quantum Feature Mapping

Converts classical medical data into quantum states.

---

# Dataset Used

Breast Cancer Wisconsin Dataset from Scikit-learn.

Dataset Information:
- Real medical dataset
- 569 patient samples
- 30 diagnostic features
- Binary classification problem

Classes:
- 0 → Malignant
- 1 → Benign

---

# Dimensionality Reduction

The original dataset contains 30 medical features.

Since current quantum systems have limited qubit capacity, Principal Component Analysis (PCA) is used to reduce:

```text
30 Features → 2 Quantum-Compatible Features
```

This allows efficient quantum processing using a 2-qubit quantum circuit.

---

# Project Workflow

```text
Medical Dataset
        ↓
Data Normalization
        ↓
PCA Dimensionality Reduction
        ↓
Quantum Feature Encoding
        ↓
Variational Quantum Circuit
        ↓
Quantum Classification
        ↓
Prediction Output
```

---

# Project Structure

```text
QuantumMLProject/
│
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocess.py
│
├── model/
│   ├── __init__.py
│   ├── quantum_model.py
│   └── train.py
│
├── utils/
│   ├── __init__.py
│   └── metrics.py
│
└── visualization/
    ├── __init__.py
    └── confusion_matrix_plot.py
```

---

# Installation Guide (macOS)

## 1. Clone Repository

```bash
git clone https://github.com/piyushshewale/QuantumMLProject.git
```

---

## 2. Enter Project Directory

```bash
cd QuantumMLProject
```

---

## 3. Create Virtual Environment

```bash
python3 -m venv qml_env
```

---

## 4. Activate Virtual Environment

```bash
source qml_env/bin/activate
```

---

## 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# Running the Project

```bash
python3 main.py
```

---

# Expected Output

The system displays:

- Dataset information
- Quantum circuit structures
- Model training process
- Accuracy score
- Classification report
- Confusion matrix
- Interactive cancer prediction system

---

# Example User Input

```text
Enter Mean Radius : 14
Enter Mean Texture: 20
```

Example Output:

```text
Benign Tumor Detected
Cancer Presence: NO
```

---

# Confusion Matrix

The project visualizes model performance using a confusion matrix heatmap.

The confusion matrix shows:
- Correct cancer detections
- Correct benign detections
- False positives
- False negatives

---

# Accuracy Formula

:contentReference[oaicite:0]{index=0}

---

# Educational Value

This project demonstrates:
- Quantum Machine Learning
- Healthcare AI
- Quantum-enhanced classification
- Data preprocessing
- PCA dimensionality reduction
- Quantum circuit training
- Medical data analysis

---

# Future Improvements

Possible future upgrades:
- Real quantum hardware execution
- Hybrid classical-quantum AI
- Quantum Neural Networks (QNN)
- Advanced medical datasets
- GUI-based application
- Cloud deployment

---

# Important Disclaimer

This project is intended for:
- Educational purposes
- Research demonstration
- Academic learning

This is NOT a real medical diagnostic system.

---

# Author

Piyush Anandrao Shewale

---

# Copyright Notice

Copyright (c) 2026 Piyush Anandrao Shewale

All rights reserved.

This project was developed for educational and academic purposes.

Unauthorized commercial redistribution or claiming this work without proper credit is discouraged.

---

# License

Educational Use License