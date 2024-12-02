import json
import os
import itertools
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import timm
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

# Load all models
model_names = {
    "facebook/vit-mae-large": "facebook/vit-mae-large",
    "google/vit-large-patch16-224": "google/vit-large-patch16-224",
    "microsoft/beit-large-patch16-512": "microsoft/beit-large-patch16-512",
    "facebook/dinov2-large-imagenet1k-1-layer": "facebook/dinov2-large-imagenet1k-1-layer",
    "vit_large_patch16_224.augreg_in21k": "vit_large_patch16_224.augreg_in21k",
}

# Selected layers based on analysis with 10 image of three classes
# best_layers = {
#     "facebook/vit-mae-large": [21, 22, 15],
#     "google/vit-large-patch16-224": [1, 3, 8],
#     "microsoft/beit-large-patch16-512": [15, 18, 19],
#     "facebook/dinov2-large-imagenet1k-1-layer": [15, 18, 23],
#     "vit_large_patch16_224.augreg_in21k": [1, 3, 20],
# }

# Selected layers based on analysis with 15 image of five classes
best_layers = {
    "facebook/vit-mae-large": [15, 19, 21],
    "google/vit-large-patch16-224": [0, 4, 5],
    "microsoft/beit-large-patch16-512": [5, 18, 21],
    "facebook/dinov2-large-imagenet1k-1-layer": [12, 19, 20],
    "vit_large_patch16_224.augreg_in21k": [1, 19, 21],
}


# Initialize transformers and timm models
models = {}
feature_extractors = {}

for name, model_name in model_names.items():
    print(f'Initializing {name}')
    if name == "vit_large_patch16_224.augreg_in21k":  # Use timm for this specific model
        models[name] = timm.create_model(model_name, pretrained=True)
    else:
        models[name] = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        feature_extractors[name] = AutoImageProcessor.from_pretrained(model_name)

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


# Extract features from specific layers
def extract_features(model, inputs, layer_indices):
    with torch.no_grad():
        if isinstance(model, timm.models.vision_transformer.VisionTransformer):
            outputs = model(inputs["pixel_values"])
            features = [outputs.mean(dim=1).cpu().numpy().reshape(-1, 1) for _ in layer_indices]
        else:
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            features = [hidden_states[layer_idx].mean(dim=1).cpu().numpy() for layer_idx in layer_indices]
    return np.concatenate(features, axis=1)


# Few-shot evaluation
def few_shot_evaluation(features, labels, shot_count):
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, stratify=labels, train_size=shot_count
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Main processing
def evaluate_layer_combinations(models, feature_extractors, best_layers, image_dir, shot_count):
    results = {}
    inputs, labels, _ = preprocess_images(image_dir, None, 224)  # Assuming unified size for simplicity

    # Evaluate each model's top three layers
    for name, model in models.items():
        print(f"Evaluating {name}'s top layers...")
        feature_extractor = feature_extractors.get(name, None)
        layer_indices = best_layers[name]
        features = extract_features(model, inputs, layer_indices)
        acc = few_shot_evaluation(features, labels.numpy(), shot_count)
        results[(name,)] = acc

    # Evaluate combinations of models
    model_names = list(models.keys())
    for r in range(2, len(model_names) + 1):
        for combination in itertools.combinations(model_names, r):
            print(f"Evaluating combination: {combination}")
            combined_features = []
            for name in combination:
                model = models[name]
                feature_extractor = feature_extractors.get(name, None)
                layer_indices = best_layers[name]
                features = extract_features(model, inputs, layer_indices)
                combined_features.append(features)
            combined_features = np.concatenate(combined_features, axis=1)
            acc = few_shot_evaluation(combined_features, labels.numpy(), shot_count)
            results[combination] = acc

    return results


# Set up directories and parameters
image_dir = "data"
shot_count = 5  # Few-shot samples per class

# Run evaluation
results = evaluate_layer_combinations(models, feature_extractors, best_layers, image_dir, shot_count)

output = {}
# Display results
for combination, acc in results.items():
    print(f"Combination {combination}: Accuracy = {acc:.4f}")
    output[str(combination)] = acc  # Convert combination tuple to string for JSON compatibility

# Save results as a JSON file
with open('layer_combinations_results.json', 'w') as json_file:
    json.dump(output, json_file, indent=4)

print("Results have been saved to 'results.json'")
