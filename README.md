🧠 Facial Expression Recognition using Deep Convolutional Neural Networks

A deep learning project that detects **human facial emotions** from grayscale images using a custom-built **Convolutional Neural Network (CNN)** trained on the **FER2013 dataset**.

---

📌 Overview

Facial expressions are a powerful indicator of human emotions. This project builds an end-to-end facial expression recognition system that classifies facial images into seven emotional categories:

Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

The complete workflow is implemented in Google Colab and demonstrates the entire pipeline from dataset loading and preprocessing to model training and feature visualization.

---

🗂 Dataset

FER2013 Dataset

* Image size: `48 × 48` (grayscale)
* Total samples: ~35,000 facial images
* Classes: 7 emotion categories
* Format: CSV file with pixel values as strings

---

⚙️ Model Architecture

The CNN architecture used in this project:

```text
Input: 48×48×1

Conv2D (32 filters, 3×3) → ReLU → MaxPooling  
Conv2D (64 filters, 3×3) → ReLU → MaxPooling  
Conv2D (128 filters, 3×3) → ReLU → MaxPooling  
Conv2D (256 filters, 3×3) → ReLU → MaxPooling  

Flatten  
Dense (1000 units, ReLU)  
Dense (7 units, Softmax)
```

This layered structure enables the model to learn:

* Low-level features (edges, textures)
* Mid-level facial parts (eyes, mouth)
* High-level emotional patterns

---

🚀 How to Run

1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Navigate to Datathe set Directory

```python
%cd /content/drive/MyDrive/Facial_Expression/
```

3. Load Dataset

```python
import pandas as pd
df = pd.read_csv('fer2013.csv')
```

4. Run All Cells

Execute the notebook sequentially to train the CNN and visualize the intermediate feature maps.

---

📊 Outputs

* Trained a CNN capable of classifying facial emotions
* Visualization of:

  * Convolution feature maps
  * ReLU activation outputs
  * Max-pooled feature representations
* End-to-end emotion prediction pipeline

---

🧠 What This Project Demonstrates

| Concept               | Description                                             |
| --------------------- | ------------------------------------------------------- |
| Convolution Filters   | Learn spatial facial features automatically             |
| Pooling Layers        | Reduce dimensionality while preserving emotion patterns |
| Deep Feature Learning | Builds hierarchical understanding of faces              |
| CNN Visualization     | Understand what the network sees at each stage          |
| FER Pipeline          | Complete emotion detection system                       |

---

📈 Sample Results

* Input size: `48×48×1`
* CNN Depth: 4 convolution layers
* Output classes: 7 emotions
* Feature compression: 48×48 → 3×3 feature maps
* Final classification using Softmax

---

🔮 Future Enhancements

* Add Dropout & Batch Normalization for regularization
* Implement data augmentation
* Evaluate performance using confusion matrix & classification report
* Apply transfer learning with MobileNet / ResNet
* Real-time webcam emotion detection

---

📜 License

This project is released for academic and educational use.

---

This repository demonstrates how deep learning converts pixel-level facial data into meaningful emotional intelligence — forming the basis of next-generation human–computer interaction systems.
