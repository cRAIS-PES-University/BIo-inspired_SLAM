import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def preprocess_images(images, target_size=(128, 128)):
    processed_images = []
    for img in images:
        resized = cv2.resize(img, target_size)
        processed_images.append(resized)
    return processed_images

def calculate_frame_differences(images):
    differences = []
    for i in range(1, len(images)):
        diff = cv2.absdiff(images[i-1], images[i])
        differences.append(diff)
    return differences

def normalize_differences(differences):
    normalized_diffs = []
    for diff in differences:
        mean_val = np.mean(diff)
        std_val = np.std(diff)
        if std_val == 0:
          std_val = 1e-10  # Add epsilon to avoid division by zero
        normalized = (diff - mean_val) / std_val
        normalized_diffs.append(normalized)
    return normalized_diffs

def convert_to_spike_trains(normalized_diffs, threshold=1.0):
    spike_trains = []
    for diff in normalized_diffs:
        spikes = np.where(diff > threshold, 1, 0)
        spike_trains.append(spikes)
    spike_trains = np.array(spike_trains)  # Ensure output is a numpy array
    if spike_trains.ndim == 1:
        spike_trains = spike_trains[np.newaxis, :]  # Ensure 2D even for single spike train
    return spike_trains