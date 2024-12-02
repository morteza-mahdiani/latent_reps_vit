import itertools
import os

import timm
import torch
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
from timm import create_model
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

# Models to analyze
models = {
    "facebook/vit-mae-large": AutoModel.from_pretrained("facebook/vit-mae-large", output_hidden_states=True),
    "google/vit-large-patch16-224": AutoModel.from_pretrained("google/vit-large-patch16-224",
                                                              output_hidden_states=True),
    "microsoft/beit-large-patch16-512": AutoModel.from_pretrained("microsoft/beit-large-patch16-512",
                                                                  output_hidden_states=True),
    "facebook/dinov2-large-imagenet1k-1-layer": AutoModel.from_pretrained("facebook/dinov2-large-imagenet1k-1-layer",
                                                                          output_hidden_states=True),
    "vit_large_patch16_224.augreg_in21k": timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True),
}

# Select layers based on analysis
best_layers = {
    "facebook/vit-mae-large": [21, 22, 15],
    "google/vit-large-patch16-224": [1, 3, 8],
    "microsoft/beit-large-patch16-512": [15, 18, 19],
    "facebook/dinov2-large-imagenet1k-1-layer": ["encoder.layer.15.mlp.fc2",
                                                 "encoder.layer.18.mlp.fc2",
                                                 "encoder.layer.23.mlp.fc2"],
    "vit_large_patch16_224.augreg_in21k": [1, 3, 20]
}


# Preprocess images for Hugging Face models
def preprocess_images_hf(image_dir, feature_extractor, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor()
    ])
    images = []
    labels = []
    class_names = os.listdir(image_dir)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = Image.open(file_path).convert("RGB")
            image = transform(image)
            images.append(image.numpy())
            labels.append(label)

    images = torch.tensor(np.array(images))
    inputs = feature_extractor(images=list(images), return_tensors="pt")
    return inputs, torch.tensor(labels), class_names


# Preprocess images for DINOv2
def preprocess_images_dino(image_dir, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    images = []
    labels = []
    class_names = os.listdir(image_dir)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(image_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = Image.open(file_path).convert("RGB")
            image = transform(image)
            images.append(image)
            labels.append(label)

    return torch.stack(images), torch.tensor(labels), class_names


# Extract features from specific layers
def extract_features_hf(model, inputs, layer_indices):
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        combined_features = torch.cat([hidden_states[i].mean(dim=1) for i in layer_indices], dim=1)
    return combined_features


# Extract features from DINOv2
def extract_features_dino(model, inputs, layer_names):
    """
    Extract features from specific layers of the DINOv2 model using named_modules.
    Handles outputs with sequence dimension.
    """
    features = []

    # Hook function to capture the output of the specified layer
    def hook(module, input, output):
        # Check dimensionality of the output
        if len(output.shape) == 4:  # If 4D (e.g., batch_size x channels x height x width)
            features.append(output.mean(dim=[2, 3]))  # Global average pooling on spatial dimensions
        elif len(output.shape) == 3:  # If 3D (e.g., batch_size x sequence_length x feature_dim)
            features.append(output.mean(dim=1))  # Mean pooling over the sequence dimension
        elif len(output.shape) == 2:  # If 2D (e.g., batch_size x feature_dim)
            features.append(output)  # Directly append as no pooling is needed
        else:
            raise ValueError(f"Unexpected output dimensions: {output.shape}")

    # Register hooks for specified layer names
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(hook))

    # Pass inputs through the model
    with torch.no_grad():
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate features from all specified layers
    return torch.cat(features, dim=1)


# Few-shot evaluation
def few_shot_evaluation(features, labels, shot_count):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels, train_size=shot_count
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Evaluate all models
def evaluate_models(models, best_layers, image_dir, shot_count):
    accuracies = {}
    for model_name, model in tqdm(models.items(), desc="Evaluating Models"):
        if "dinov2" in model_name:  # Handle DINOv2
            inputs, labels, class_names = preprocess_images_dino(image_dir)
            features = extract_features_dino(model, inputs, best_layers[model_name])
        else:  # Handle Hugging Face models
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            image_size = 512 if "beit" in model_name else 224
            inputs, labels, class_names = preprocess_images_hf(image_dir, feature_extractor, image_size)
            features = extract_features_hf(model, inputs, best_layers[model_name])

        # Few-shot evaluation
        accuracy = few_shot_evaluation(features.numpy(), labels.numpy(), shot_count)
        accuracies[model_name] = accuracy

    return accuracies


# Pairwise evaluation
def evaluate_model_pairs(models, best_layers, image_dir, shot_count):
    """
    Evaluate pairwise combinations of models for few-shot learning.
    """
    pair_accuracies = {}
    model_names = list(models.keys())

    # Evaluate each pair of models
    for model1_name, model2_name in itertools.combinations(model_names, 2):
        print(f"Evaluating pair: {model1_name} and {model2_name}")

        # Process first model
        model1 = models[model1_name]
        if "dinov2" in model1_name:  # Handle DINOv2
            inputs1, labels1, class_names = preprocess_images_dino(image_dir)
            features1 = extract_features_dino(model1, inputs1, best_layers[model1_name])
        else:  # Handle Hugging Face models
            feature_extractor1 = AutoFeatureExtractor.from_pretrained(model1_name)
            image_size1 = 512 if "beit" in model1_name else 224
            inputs1, labels1, class_names = preprocess_images_hf(image_dir, feature_extractor1, image_size1)
            features1 = extract_features_hf(model1, inputs1, best_layers[model1_name])

        # Process second model
        model2 = models[model2_name]
        if "dinov2" in model2_name:  # Handle DINOv2
            inputs2, labels2, _ = preprocess_images_dino(image_dir)
            features2 = extract_features_dino(model2, inputs2, best_layers[model2_name])
        else:  # Handle Hugging Face models
            feature_extractor2 = AutoFeatureExtractor.from_pretrained(model2_name)
            image_size2 = 512 if "beit" in model2_name else 224
            inputs2, labels2, _ = preprocess_images_hf(image_dir, feature_extractor2, image_size2)
            features2 = extract_features_hf(model2, inputs2, best_layers[model2_name])

        # Ensure labels are consistent
        assert torch.equal(labels1, labels2), "Labels mismatch between models!"

        # Combine features from both models
        combined_features = torch.cat((features1, features2), dim=1)

        # Few-shot learning with combined features
        accuracy = few_shot_evaluation(combined_features.numpy(), labels1.numpy(), shot_count)
        pair_accuracies[(model1_name, model2_name)] = accuracy

    return pair_accuracies


# Set parameters
image_dir = "data"  # Path to dataset
shot_count = 5  # Few-shot samples per class

mode = 'pairwise'

if mode == 'pairwise':
    # Run pairwise evaluation
    pair_accuracies = evaluate_model_pairs(models, best_layers, image_dir, shot_count)

    # Plot pairwise results
    pairs = [" + ".join(pair) for pair in pair_accuracies.keys()]
    accuracies = list(pair_accuracies.values())

    plt.figure(figsize=(12, 6))
    plt.bar(pairs, accuracies, color="blue")
    plt.xlabel("Model Pairs")
    plt.ylabel("Few-Shot Accuracy")
    plt.title("Few-Shot Learning Accuracy Across Model Pairs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if mode == 'single':
    # Run evaluation
    accuracies = evaluate_models(models, best_layers, image_dir, shot_count)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color="blue")
    plt.xlabel("Models")
    plt.ylabel("Few-Shot Accuracy")
    plt.title("Few-Shot Learning Accuracy Across Models")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
