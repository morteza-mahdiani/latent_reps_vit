import os
import torch
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


# Load the models
model_names = {
    "beit": "microsoft/beit-large-patch16-512",
    "vit": "google/vit-large-patch16-224",
    "mae": "facebook/vit-mae-large"
}

models = {
    name: AutoModel.from_pretrained(model, output_hidden_states=True)
    for name, model in model_names.items()
}

feature_extractors = {
    name: AutoFeatureExtractor.from_pretrained(model)
    for name, model in model_names.items()
}

# Define layers to analyze (24 for ViT-based architectures)
layers_to_analyze = list(range(24))


# Helper function to preprocess images
def preprocess_images(image_dir, feature_extractor, image_size):
    images = []
    labels = []
    class_names = os.listdir(image_dir)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = Image.open(file_path).convert("RGB")
            image = transform(image)
            images.append(image.numpy())
            labels.append(label)

    images = torch.tensor(np.array(images))
    labels = torch.tensor(labels)

    inputs = feature_extractor(images=list(images), return_tensors="pt")
    return inputs, labels, class_names


# Extract features from a specific layer
def extract_features(model, inputs, layer_idx):
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        features = hidden_states[layer_idx].mean(dim=1)
    return features


# Few-shot evaluation
def few_shot_evaluation(features, labels, shot_count):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels, train_size=shot_count
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Main processing and evaluation for all models
def evaluate_all_models(models, feature_extractors, layers, image_dir, shot_count):
    accuracies = {name: [] for name in models.keys()}
    class_names = None

    for name, model in models.items():
        feature_extractor = feature_extractors[name]
        image_size = 512 if "beit" in name else 224

        # Preprocess images
        inputs, labels, class_names = preprocess_images(image_dir, feature_extractor, image_size)

        for layer_idx in tqdm(layers, desc=f"Processing Layers for {name}"):
            # Extract features
            features = extract_features(model, inputs, layer_idx)

            # Few-shot evaluation
            acc = few_shot_evaluation(features.numpy(), labels.numpy(), shot_count)
            accuracies[name].append(acc)

    return accuracies, class_names


# Set up directories and parameters
image_dir = "data"
shot_count = 5  # Few-shot samples per class

# Evaluate models
accuracies, class_names = evaluate_all_models(models, feature_extractors, layers_to_analyze, image_dir, shot_count)

# Plot results for each model
plt.figure(figsize=(14, 8))
for name, acc in accuracies.items():
    plt.plot(range(len(layers_to_analyze)), acc, label=name)
plt.xlabel("Layer Index")
plt.ylabel("Few-Shot Accuracy")
plt.title("Few-Shot Learning Performance Across Layers for All Models")
plt.legend()
plt.grid()
plt.show()

# Suggest candidate layers based on top performance
top_layers = {}
for name, acc in accuracies.items():
    best_layer_indices = sorted(range(len(acc)), key=lambda i: acc[i], reverse=True)[:3]  # Top 3 layers
    top_layers[name] = best_layer_indices

print("Top candidate layers for each model:")
for name, layers in top_layers.items():
    print(f"{name}: {layers}")
