# Comparative Analysis of Convolutional and Transformer Architectures for Real-Time Fire Detection

**Author:** Puttala Jeevan Kumar – Computer Vision & AI/ML Researcher  
**Date:** March 2026  
*Full Transparent Comparative Analysis of Convolutional and Transformer Architectures for Real-Time Fire Detection*



## Abstract
This study presents a rigorous empirical comparison between a Convolutional Neural Network (YOLOv8s) and a Vision Transformer (RT-DETR-L) for the specialized task of real-time fire and smoke detection. The objective was to evaluate the mathematical trade-offs between localized inductive bias and global self-attention mechanisms in edge-deployment scenarios (e.g., autonomous drones and automotive vision systems). Experimental results demonstrate a clear architectural divergence: the lightweight CNN achieved superior recall and mean Average Precision (mAP@50 = 0.867), proving highly parameter-efficient for capturing localized boundary anomalies. Conversely, the Vision Transformer exhibited superior background suppression, reducing False Positives by 33% and achieving the highest overall Precision (0.896). This analysis provides a calibrated deployment framework for hardware integration based on operational risk tolerance.

## 1. Introduction & Objectives
Automated edge-based fire detection systems require algorithms capable of scale invariance, high-speed inference, and extreme robustness to visual noise (e.g., fog, occlusion, and lighting shifts). Traditional object detection has been dominated by Convolutional Neural Networks (CNNs). However, the advent of Vision Transformers (ViTs) introduces self-attention mechanisms capable of understanding global image context without anchor-box heuristics.

This project benchmarks YOLOv8s (Small) against RT-DETR-L (Large) to determine the optimal architecture for real-world environmental monitoring, answering a critical engineering question: Does the global contextual awareness of a heavy Transformer outweigh the highly efficient, localized feature extraction of a CNN in specialized anomaly detection?

## 2. Dataset Geometry & Augmentation
To prevent model center-bias and ensure real-world robustness, the dataset and preprocessing pipelines were strictly controlled and aggressively augmented.

### 2.1 Spatial and Semantic Distribution
The dataset maintained strict equilibrium across critical classes: fire, smoke, and no-fire. The robust inclusion of the no-fire negative space is critical for teaching the model boundary discrimination and reducing hallucinated bounding boxes. Coordinate distribution analysis confirmed zero central clustering, forcing the network to scan peripheral fields of view.

### 2.2 Augmentation Pipeline
To simulate challenging aerial and automotive environments, dynamic data augmentation was applied during the 200-epoch training loop:
* **4-Way Mosaic Stitching:** Combined multiple images into a single training frame to teach scale invariance.
* **Luminance & Spectral Jittering:** Applied HSV adjustments (Saturation: 0.7, Value: 0.4) to simulate severe weather, overcast skies, and high-glare lighting conditions.

## 3. Architectural Methodology
Both models were trained under identically constrained hardware parameters (NVIDIA RTX 4060, 640x640 resolution, batch size 8) to evaluate two fundamentally different mathematical approaches to computer vision.

### 3.1 Convolutional Inductive Bias (YOLOv8s)
YOLOv8s relies on a CNN backbone. Its architecture inherently assumes that neighboring pixels are related (inductive bias). This allows for rapid, parameter-efficient learning, particularly for highly localized phenomena like expanding flames.

### 3.2 Global Self-Attention (RT-DETR-L)
Real-Time Detection Transformer (RT-DETR) processes the image as a sequence of patches. It utilizes a hybrid encoder and multi-scale self-attention to understand the entire image simultaneously, relating smoke in the top-left corner to a fire in the bottom-right corner without relying on localized sliding windows.

## 4. Comparative Results & Metrics

### 4.1 The Stochastic Convergence Gap
Analysis of the training curves revealed a distinct behavioral difference between the architectures. YOLOv8s demonstrated a smooth, logarithmic ascent, reaching stability within the first 50 epochs due to its inherent spatial bias. In contrast, RT-DETR-L exhibited a highly volatile, stochastic convergence trajectory. Lacking inductive bias, the Transformer required significantly more epochs to mathematically stabilize its spatial relationships.

### 4.2 Quantitative Benchmarking
Despite its significantly smaller parameter count, the CNN demonstrated superior overall localization (Recall), while the Transformer excelled in absolute certainty (Precision).

### 4.3 False Positive Suppression
The defining victory for the Vision Transformer was its background suppression. Evaluation of the normalized confusion matrices revealed that the Transformer's global attention mechanism reduced false alarms by an impressive 33% compared to the CNN.

*Normalized Confusion Matrix illustrating false positive suppression. The evaluation reveals that the Transformer's global attention mechanism successfully reduced false alarms by an impressive 33% compared to the CNN.*

## 5. Real-Time Inference & Video Analysis
Static metrics do not fully capture edge-deployment viability. Both models were tested on unseen, dynamic drone footage to evaluate temporal stability and inference speed.

* **YOLOv8s:** Exhibited rapid bounding box generation with high recall, successfully tracking rapidly expanding brush fires even when partially obscured.
* **RT-DETR-L:** Exhibited highly stable bounding boxes with zero anchor-box jitter, cleanly isolating the fire core without being distracted by peripheral thermal noise.

**Video Link:** [https://youtu.be/ynnPtxrfhC0](https://youtu.be/ynnPtxrfhC0)

*Real-time inference on unseen dynamic drone footage for smoke detection. RT-DETR-L exhibited highly stable bounding boxes with zero anchor-box jitter, cleanly isolating the core without being distracted by peripheral thermal noise.*

*Real-time inference on unseen dynamic drone footage for fire detection. YOLOv8s exhibited rapid bounding box generation with high recall, successfully tracking rapidly expanding brush fires even when partially obscured.*

## 6. Conclusion & Deployment Logic
The selection of the final architecture depends entirely on the operational risk profile of the deployment environment.

1. **Maximum Detection & Efficiency (YOLOv8s):** For environments where a missed fire is catastrophic and false alarms are an acceptable operational cost (e.g., early-warning forest monitoring), YOLOv8s is the superior architecture. It is highly parameter-efficient and suitable for low-power edge devices. Optimal deployment calibration dictates an F1 confidence threshold of ~0.29.
2. **Alert Suppression & Accuracy (RT-DETR-L):** For automated dispatch systems or autonomous vehicle perception where false positives carry high financial or logistical costs, RT-DETR-L provides superior operational stability. Optimal deployment calibration dictates an F1 confidence threshold of ~0.22.
