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
beit_model_name = "microsoft/beit-large-patch16-512"
vit_model_name = "google/vit-large-patch16-224"

beit_model = AutoModel.from_pretrained(beit_model_name, output_hidden_states=True)
vit_model = AutoModel.from_pretrained(vit_model_name, output_hidden_states=True)

# Load the feature extractors
beit_feature_extractor = AutoFeatureExtractor.from_pretrained(beit_model_name)
vit_feature_extractor = AutoFeatureExtractor.from_pretrained(vit_model_name)

# Define layers to analyze
layers_to_analyze_vit = [
    f"encoder.layer.{i}.output.dense" for i in range(24)
]


# Helper function to preprocess images
def preprocess_images(image_dir, feature_extractor, image_size):
    """
    Preprocess images for the feature extractor with resizing and grayscale conversion.
    """
    images = []
    labels = []
    class_names = os.listdir(image_dir)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.Resize((image_size, image_size)),  # Resize to model-specific dimensions
        transforms.ToTensor()  # Convert to tensor
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
        features = hidden_states[layer_idx].mean(dim=1)  # Mean pooling
    return features


# Few-shot evaluation
def few_shot_evaluation(features, labels, shot_count):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels, train_size=shot_count
    )

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Main processing and evaluation
def evaluate_model_combined(beit_model, vit_model, beit_feature_extractor, vit_feature_extractor, layers, image_dir,
                            shot_count):
    """
    Evaluate BEiT, ViT, and their combined features for few-shot learning.
    """
    # Preprocess images for both BEiT and ViT
    beit_inputs, labels, class_names = preprocess_images(image_dir, beit_feature_extractor, image_size=512)
    vit_inputs, _, _ = preprocess_images(image_dir, vit_feature_extractor, image_size=224)

    beit_accuracies = []
    vit_accuracies = []
    combined_accuracies = []

    for layer_idx, layer_name in enumerate(tqdm(layers, desc="Processing Layers")):
        # Extract features from BEiT
        beit_features = extract_features(beit_model, beit_inputs, layer_idx)

        # Extract features from ViT
        vit_features = extract_features(vit_model, vit_inputs, layer_idx)

        # Concatenate BEiT and ViT features
        combined_features = torch.cat((beit_features, vit_features), dim=1)

        # Few-shot learning for each feature set
        beit_acc = few_shot_evaluation(beit_features.numpy(), labels.numpy(), shot_count)
        vit_acc = few_shot_evaluation(vit_features.numpy(), labels.numpy(), shot_count)
        combined_acc = few_shot_evaluation(combined_features.numpy(), labels.numpy(), shot_count)

        # Store accuracies
        beit_accuracies.append(beit_acc)
        vit_accuracies.append(vit_acc)
        combined_accuracies.append(combined_acc)

    return beit_accuracies, vit_accuracies, combined_accuracies, class_names


# Set up directories and parameters
image_dir = "data"
shot_count = 5  # Few-shot samples per class

# Evaluate models
beit_accuracies, vit_accuracies, combined_accuracies, class_names = evaluate_model_combined(
    beit_model, vit_model, beit_feature_extractor, vit_feature_extractor, layers_to_analyze_vit, image_dir, shot_count
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(len(layers_to_analyze_vit)), beit_accuracies, label="BEiT")
plt.plot(range(len(layers_to_analyze_vit)), vit_accuracies, label="ViT")
plt.plot(range(len(layers_to_analyze_vit)), combined_accuracies, label="BEiT + ViT")
plt.xlabel("Layer Index")
plt.ylabel("Few-Shot Accuracy")
plt.title("Few-Shot Learning Performance Across Layers")
plt.legend()
plt.grid()
plt.show()
