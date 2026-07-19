# 🧠 Brain Tumor MRI Classification System

An end-to-end Deep Learning pipeline engineered to classify brain anomalies from MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary tumors, or Normal (No Tumor)**.

## 🚀 Key Features
* **Multi-Model Evaluation:** Conducted a rigorous comparative evaluation between three state-of-the-art convolutional architectures: **InceptionV3**, **VGG19**, and **ResNet152V2**.
* **Production-Ready Deployment:** Packaged the pipeline into a modular, object-oriented Python inference class capable of dynamic image scaling and real-time validation.
* **Interactive Web UI:** Integrated a functional front-end web application interface using Gradio to handle raw user file ingestion and live confidence scoring.

## 📊 Experimental Results & Comparison
The models were trained and tested on a curated dataset of over 5,700 MRI images using a standard 80/10/10 split.

| Model Architecture | Validation Accuracy | Macro Avg F1-Score |
| :--- | :---: | :---: |
| VGG19 | 90.33% | 0.91 |
| InceptionV3 | 92.79% | 0.92 |
| **ResNet152V2 (Optimal)** | **94.02%** | **0.96** |

*The **ResNet152V2** architecture emerged as the top performer, leveraging residual skip-connections to effectively map structural anomalies and achieve near-perfect individual test predictions.*

## 🛠️ Tech Stack
* **Frameworks:** TensorFlow / Keras, NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Interface & Deployment:** Gradio
*
