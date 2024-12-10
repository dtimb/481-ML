import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import rasterio
from PIL import Image
import os

IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3  # RGB images

def load_data(image_dir, label_dir):
    images = []
    masks = []

    # Load images
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.tif'):
            img_path = os.path.join(image_dir, img_name)
            with rasterio.open(img_path) as src:
                img = src.read()  # Read the image as a NumPy array
                img = np.transpose(img, (1, 2, 0))  # Change from (bands, height, width) to (height, width, bands)

                # Convert to RGB by discarding the alpha channel if it exists
                if img.shape[2] == 4:  # Check if it has an alpha channel
                    img = img[:, :, :3]  # Keep only the RGB channels

                img = Image.fromarray(img.astype(np.uint8)).convert('RGB')  # Convert to RGB
                img = img.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize if necessary
                images.append(np.array(img))

    # Load masks
    for mask_name in os.listdir(label_dir):
        if mask_name.endswith('.tif'):
            mask_path = os.path.join(label_dir, mask_name)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Read the first band
                mask = Image.fromarray(mask.astype(np.uint8))  # Convert to PIL Image
                mask = mask.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize if necessary
                masks.append(np.array(mask))

    # Convert masks to the required shape (height, width, 1)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)  # Add a channel dimension

    return np.array(images), masks


def load_pretrained_models():
    vgg16 = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    return vgg16, vgg19, resnet

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def extract_features(model, layer_name, images, target_size=512):
    model_layer = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = model_layer.predict(images, verbose=1)
    
    # Apply Global Average Pooling to reduce spatial dimensions
    pooled_features = GlobalAveragePooling2D()(features)

    # Apply Dense layer to reduce the number of channels to the target size
    if pooled_features.shape[-1] != target_size:
        dense_layer = Dense(target_size, activation='relu')
        pooled_features = dense_layer(pooled_features)
    
    return pooled_features

def concatenate_features(features_vgg16, features_vgg19, features_resnet, method='avg'):
    # Print the shapes for debugging purposes
    print("Shape of VGG16 pooled features:", features_vgg16.shape)
    print("Shape of VGG19 pooled features:", features_vgg19.shape)
    print("Shape of ResNet pooled features:", features_resnet.shape)

    if not (features_vgg16.shape == features_vgg19.shape == features_resnet.shape):
        raise ValueError("Feature shapes do not match among the three models after pooling and dense layer adjustment.")

    # Perform the requested concatenation method
    if method == 'avg':
        combined_features = np.mean([features_vgg16, features_vgg19, features_resnet], axis=0)
    elif method == 'min':
        combined_features = np.min([features_vgg16, features_vgg19, features_resnet], axis=0)
    elif method == 'max':
        combined_features = np.max([features_vgg16, features_vgg19, features_resnet], axis=0)
    else:
        raise ValueError("Method must be 'min', 'avg', or 'max'")
    return combined_features

def apply_dimensionality_reduction(features, method='pca', n_components=50):
    if method == 'pca':
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
    elif method == 'tsne':
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_features = tsne.fit_transform(features)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    return reduced_features

def train_svm_classifier(features, labels):
    svm = SVC(kernel='linear')
    svm.fit(features, labels)
    return svm

def evaluate_model(svm, features, true_labels):
    predictions = svm.predict(features)
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    return accuracy, report

def create_image_labels(masks):
    """
    Aggregates pixel-level masks into a single image-level label.
    For binary classification, consider if the image contains a significant amount of positive pixels.
    """
    image_labels = []
    threshold = 0.5  # Example threshold: 50% of the pixels in the image must be positive to be labeled as 1

    for mask in masks:
        # Compute the fraction of pixels that are positive in the mask
        positive_fraction = np.mean(mask > 0)
        image_labels.append(1 if positive_fraction > threshold else 0)

    return np.array(image_labels)

def main():
    # Step 1: Load models
    vgg16, vgg19, resnet = load_pretrained_models()

    dataset_path = ' ' # adjust dataset path
    mask_path = ' ' # adjust mask path
    input_shape = (512, 512)

    # Step 2: Load training images and masks
    train_images, train_masks = load_data(dataset_path, mask_path)
    
    # Normalize images to range [0, 1]
    train_images = train_images / 255.0

    # Create image-level labels from the masks
    true_labels = create_image_labels(train_masks)
    print("Shape of true labels:", true_labels.shape)

    # Step 3: Extract Features
    features_vgg16 = extract_features(vgg16, 'block5_pool', train_images)
    features_vgg19 = extract_features(vgg19, 'block5_pool', train_images)
    features_resnet = extract_features(resnet, 'conv5_block3_out', train_images)

    # Step 4: Concatenate Features (min, max, avg example)
    combined_features = concatenate_features(features_vgg16, features_vgg19, features_resnet, method='avg')

    # Step 5: Dimensionality Reduction using PCA or T-SNE
    pca = PCA(n_components=50)
    reduced_features = pca.fit_transform(combined_features)

    # Step 6: Apply SVM on reduced features
    svm = SVC()
    svm.fit(reduced_features, true_labels)

    # Step 7: Evaluate Model
    accuracy, report = evaluate_model(svm, reduced_features, true_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(report)

    # Step 8: Save results
    np.save('reduced_features.npy', reduced_features)
    np.save('true_labels.npy', true_labels)

if __name__ == "__main__":
    main()