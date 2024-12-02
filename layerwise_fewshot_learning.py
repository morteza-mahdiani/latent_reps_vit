import json
import os
import torch
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import timm

# Load all models, including the new ones
model_names = {
    "beit": "microsoft/beit-large-patch16-512",
    "vit": "google/vit-large-patch16-224",
    "mae": "facebook/vit-mae-large",
    "dinov2": "facebook/dinov2-large-imagenet1k-1-layer",
    "vit_augreg": "vit_large_patch16_224.augreg_in21k",
}

# Initialize transformers and timm models
models = {}
feature_extractors = {}

for name, model_name in model_names.items():
    print('name of the model that is processing is ', model_name)
    if name == "vit_augreg":  # Use timm for this specific model
        models[name] = timm.create_model(model_name, pretrained=True)
    elif name == "dinov2":  # Use AutoImageProcessor for this specific model
        models[name] = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        feature_extractors[name] = AutoImageProcessor.from_pretrained(model_name)
    else:
        models[name] = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        feature_extractors[name] = AutoFeatureExtractor.from_pretrained(model_name)

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

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            # Filter for valid image files
            if os.path.splitext(file_name)[1].lower() in valid_extensions:
                try:
                    image = Image.open(file_path).convert("RGB")
                    image = transform(image)
                    images.append(image.numpy())
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    images = torch.tensor(np.array(images))
    labels = torch.tensor(labels)

    if feature_extractor is not None:
        inputs = feature_extractor(images=list(images), return_tensors="pt")
    else:  # For timm models
        inputs = {"pixel_values": images}

    return inputs, labels, class_names


# Extract features from a specific layer
# Extract features from a specific layer
def extract_features(model, inputs, layer_idx):
    with torch.no_grad():
        if isinstance(model, timm.models.vision_transformer.VisionTransformer):
            outputs = model(inputs["pixel_values"])
            # Ensure features are 2D (samples, features)
            features = outputs.mean(dim=1).cpu().numpy().reshape(-1, 1)  # Reshaping to 2D
        else:
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            # Ensure features are 2D (samples, features)
            features = hidden_states[layer_idx].mean(dim=1).cpu().numpy()
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
        print(f"Evaluating {name}...")
        feature_extractor = feature_extractors.get(name, None)
        image_size = 512 if "beit" in name or "dinov2" in name else 224

        # Preprocess images
        inputs, labels, class_names = preprocess_images(image_dir, feature_extractor, image_size)

        for layer_idx in tqdm(layers, desc=f"Processing Layers for {name}"):
            # Extract features
            features = extract_features(model, inputs, layer_idx)

            # Few-shot evaluation
            acc = few_shot_evaluation(features, labels.numpy(), shot_count)
            accuracies[name].append(acc)

    return accuracies, class_names


# Set up directories and parameters
image_dir = "data"
shot_count = 5  # Few-shot samples per class

# Evaluate models
accuracies, class_names = evaluate_all_models(models, feature_extractors, layers_to_analyze, image_dir, shot_count)

# Save results
results_file = "few_shot_results.npy"
np.save(results_file, accuracies)

# Define colors and styles for a beautiful plot
plt.figure(figsize=(14, 8))
colors = ['blue', 'green', 'red', 'orange', 'purple']
markers = ['o', 's', 'D', '^', 'v']

for (name, acc), color, marker in zip(accuracies.items(), colors, markers):
    plt.plot(
        layers_to_analyze,
        acc,
        label=name,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=8
    )

# Enhance visualization with grid, labels, and title
plt.xlabel("Layer Index", fontsize=14)
plt.ylabel("Few-Shot Accuracy", fontsize=14)
plt.title("Few-Shot Learning Performance Across Layers for All Models", fontsize=16, fontweight='bold')
plt.legend(fontsize=12, title="Models", title_fontsize=14, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot locally
plt.savefig("few_shot_learning_performance.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Suggest candidate layers based on top performance
top_layers = {}
for name, acc in accuracies.items():
    best_layer_indices = sorted(range(len(acc)), key=lambda i: acc[i], reverse=True)[:3]  # Top 3 layers
    top_layers[name] = best_layer_indices

print("Top candidate layers for each model:")
for name, layers in top_layers.items():
    print(f"{name}: {layers}")

# Save results as a JSON file
with open('top_layers.json', 'w') as json_file:
    json.dump(top_layers, json_file, indent=4)
