# 🧠 Alzheimer's Disease MRI Classifier with Transfer Learning

This project implements a deep learning pipeline for classifying brain MRI scans into four stages of Alzheimer's disease using TensorFlow and transfer learning.

---

## 🧾 Classes

- Non Demented  
- Very Mild Dementia  
- Mild Dementia  
- Moderate Dementia  

---

## 📁 Dataset

The dataset contains 34,000+ MRI scans across the four classes.  
All images were resized to **128x128** and converted to RGB.

Split:
- 70% Training
- 24% Testing
- 6% Validation

---

## 🧠 Model Architecture

1. **Feature Extractor**  
   Pretrained model `MRI_SCAN_CLASSIFIER.h5` was used. All layers were frozen.

2. **New Classification Head**
   - Flatten  
   - Dense(128, activation='relu')  
   - Dense(4, activation='softmax')

3. **Training**
   - Optimizer: `Adam(1e-4)`
   - Loss: `Sparse Categorical Crossentropy`
   - Epochs: 5

---

## 📊 Results

| Metric              | Score       |
|---------------------|-------------|
| Validation Accuracy | **100%**    |
| F2 Score (Macro)    | **0.9997**  |




---

## 📈 Loss Trend

Add your `loss_curve.png` here once you export:

```markdown
![Loss Curve](figures/loss_curve.png)


🚀 How to Use
# Clone the repo
git clone https://github.com/your-username/alzheimer-mri-classifier.git
cd alzheimer-mri-classifier

# Install dependencies
pip install -r requirements.txt

# Train or load the model, then run inference


📦 Deployment
Gradio or Streamlit-based web app coming soon...


👨‍🔬 Author
Opeyemi Bisiriyu
Chemical Engineering @ Obafemi Awolowo University
ML Engineer | AI for Health & Energy

