# Investigating Activation of Vision Transformer (ViT) Variants

## 🌟 Project Overview

This project explores the activation patterns of layers from various Vision Transformer (ViT) variants. Our primary focus is to:

1. **Few-Shot Learning Accuracy**: Analyze the accuracy of downstream tasks by leveraging layer activations in few-shot learning scenarios.
2. **Representational Similarity Analysis**: Compare the **Representational Similarity Matrices (RSMs)** of layer activations using a shape-based dataset to understand how different layers encode information.

By combining accuracy evaluation and representational analysis, this project aims to provide insights into the inner workings of ViT models across different tasks.

---

## 🚀 Objectives

1. **Few-Shot Learning Evaluation**:
   - Investigate the performance of selected layers from multiple ViT variants in downstream tasks with few-shot learning (e.g., 5-shot learning).
   - Identify which layers contribute most significantly to performance.

2. **Representational Similarity Analysis**:
   - Compute RSMs to measure activation similarity between layers within the same model and across different models.
   - Use a shape-based dataset to ensure that the models are evaluated on consistent and interpretable data.

3. **Comparative Study**:
   - Explore how different ViT architectures and pretraining methods (e.g., MAE, DINO, BEiT) affect layer representations and task performance.

---

## 🛠️ Methodology

### 1. Vision Transformer (ViT) Variants
The following ViT variants are included in the study:
- **MAE (Masked Autoencoder)**: `facebook/vit-mae-large`
- **Standard ViT**: `google/vit-large-patch16-224`
- **BEiT**: `microsoft/beit-large-patch16-512`
- **DINOv2**: `facebook/dinov2-large-imagenet1k-1-layer`
- **AugReg ViT**: `vit_large_patch16_224.augreg_in21k`

### 2. Few-Shot Learning Accuracy
- Fine-tune each layer's activations for downstream tasks.
- Evaluate using a 5-shot learning setting.
- Compare combinations of layers from different models to assess synergy.

### 3. Representational Similarity Matrices (RSMs)
- Generate RSMs for the shape-based dataset.
- Use tools like **CKA (Centered Kernel Alignment)** or cosine similarity to measure the representational overlap.

---

## 📊 Results

### 1. Few-Shot Learning
- Achieved [X]% accuracy for the best-performing layer combination.
- Detailed results can be found in the [`results/accuracy_analysis`](./results/accuracy_analysis) directory.

### 2. Representational Similarity
- Visualizations of RSMs are available in the [`results/rsm_analysis`](./results/rsm_analysis) directory.
- Key insights:
  - Layers [A] and [B] exhibit high similarity across [models].
  - Early layers tend to focus on [patterns], while deeper layers encode [features].

---

## 📁 Repository Structure

```plaintext
.
├── data/                      # Shape-based dataset
├── models/                    # Pretrained ViT models
├── src/                       # Scripts for analysis
│   ├── few_shot_learning.py   # Few-shot learning evaluation
│   ├── rsm_analysis.py        # Representational similarity analysis
│   └── utils/                 # Helper functions
├── results/                   # Results and visualizations
│   ├── accuracy_analysis/     # Few-shot learning results
│   ├── rsm_analysis/          # RSM plots and metrics
│   └── combined/              # Combined insights
├── README.md                  # Project description
└── requirements.txt           # Python dependencies
```

---

## 📦 Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- **PyTorch**: For loading models and computing activations.
- **scikit-learn**: For similarity metrics.
- **Matplotlib**: For visualizations.
- **NumPy**: For numerical computations.

---

## 📚 References

1. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," [arXiv](https://arxiv.org/abs/2010.11929).
2. He et al., "Masked Autoencoders Are Scalable Vision Learners," [arXiv](https://arxiv.org/abs/2111.06377).
3. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," [arXiv](https://arxiv.org/abs/2104.14294).

---

## ✨ Contributors

- **Your Name**: Research and implementation.
- Contributions are welcome! Feel free to submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE). 

---
