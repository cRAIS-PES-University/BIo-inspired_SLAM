#New Dataloader with validation
# dataloader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from PIL import Image
import torchvision.transforms as transforms

def events_to_voxel_grid(events, grid_size):
    """
    Convert events to a voxel grid.

    Args:
        events (np.ndarray): Structured array of events with fields 'x', 'y', 't', 'p'
        grid_size (tuple): Grid size (H, W)

    Returns:
        torch.Tensor: Voxel grid tensor [2, H, W]
    """
    H, W = grid_size
    polarity_grid = np.zeros((H, W), dtype=np.float32)
    count_grid = np.zeros((H, W), dtype=np.float32)

    # Populate the grids
    for event in events:
        x, y, p = event['x'], event['y'], event['p']
        if 0 <= x < W and 0 <= y < H:
            polarity_grid[y, x] = p
            count_grid[y, x] += 1

    voxel_grid = np.stack([polarity_grid, count_grid], axis=0)
    voxel_grid = torch.from_numpy(voxel_grid).float()
    return voxel_grid  # [2, H, W]

class DSECDataset(Dataset):
    def __init__(self, directory_path, chunk_size=500000, grid_size=(320, 640), stream='left', transform=None, mode='train'):
        """
        Args:
            directory_path (str): Path to the root directory containing sequence folders.
            chunk_size (int): Number of events per chunk.
            grid_size (tuple): Spatial dimensions for event grids (H, W).
            stream (str): 'left' or 'right' to specify the camera stream.
            transform (callable, optional): Optional transform to be applied on images.
            mode (str): 'train' or 'validation' to specify dataset mode.
        """
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.transform = transform
        self.stream = stream.lower()  # Ensure consistency ('left' or 'right')
        self.mode = mode.lower()
        assert self.mode in ['train', 'validation'], "mode should be 'train' or 'validation'"

        # Initialize list to hold file paths
        all_files = []

        # Traverse through all sequence directories
        for seq_dir in sorted(os.listdir(self.directory_path)):
            if seq_dir.startswith('.'):
                continue  # Skip hidden directories like .ipynb_checkpoints

            seq_path = os.path.join(self.directory_path, seq_dir)
            if not os.path.isdir(seq_path):
                continue  # Skip if not a directory

            # Define paths for the selected stream
            events_file = os.path.join(seq_path, 'events', self.stream, 'events.h5')
            timestamps_file = os.path.join(seq_path, 'images', self.stream, 'exposure_timestamps.txt')

            # Check if both events and timestamps files exist
            if os.path.exists(events_file) and os.path.exists(timestamps_file):
                all_files.append((events_file, timestamps_file))
            else:
                print(f"Missing events or timestamps for sequence: {seq_dir}, stream: {self.stream}")

        self.num_total_files = len(all_files)
        print(f"Found {self.num_total_files} '{self.stream}' event files.")

        # Split files into train and validation
        if self.mode == 'train':
            if self.num_total_files <= 3:
                raise ValueError("Not enough files for training. At least 4 files are required (3 for validation).")
            self.files = all_files[:-3]  # Use all except last 3 for training
        elif self.mode == 'validation':
            self.files = all_files[-3:]  # Use last 3 files for validation

        self.num_files = len(self.files)
        print(f"Using {self.num_files} files for mode '{self.mode}'.")

        # Precompute total number of chunks
        self.total_chunks = 0
        self.chunk_map = []  # List of tuples: (file_idx, chunk_idx)

        for file_idx, (events_file, timestamps_file) in enumerate(self.files):
            # Load the number of events
            with h5py.File(events_file, 'r') as f:
                num_events = len(f['events/x'])

            # Calculate the number of chunks for this file
            chunks = (num_events + self.chunk_size - 1) // self.chunk_size
            self.total_chunks += chunks

            # Map each chunk to its file and chunk index
            for chunk_idx in range(chunks):
                self.chunk_map.append((file_idx, chunk_idx))

        print(f"Total chunks in dataset: {self.total_chunks}")

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        """
        Returns:
            A dictionary containing:
                - event_voxel_grid: Tensor of shape [2, H, W]
                - I_k: Tensor of shape [C, H, W]
                - I_k_prime: Tensor of shape [C, H, W]
        """
        file_idx, chunk_idx = self.chunk_map[idx]
        events_file, timestamps_file = self.files[file_idx]

        # Calculate start and end indices for this chunk
        with h5py.File(events_file, 'r') as f:
            num_events = len(f['events/x'])
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, num_events)

            # Load events
            x = f['events/x'][start:end].astype(np.int32)
            y = f['events/y'][start:end].astype(np.int32)
            t = f['events/t'][start:end].astype(np.int64)  # Adjust if necessary
            p = f['events/p'][start:end].astype(np.int8)

            # Handle polarity conversion if needed
            p = p * 2 - 1  # Convert from {0,1} to {-1,1}

            # Create a structured array or use separate arrays
            events = np.zeros(len(x), dtype=[('x', np.int32), ('y', np.int32), ('t', np.int64), ('p', np.int8)])
            events['x'] = x
            events['y'] = y
            events['t'] = t
            events['p'] = p

        # Convert events to voxel grid
        voxel_grid = events_to_voxel_grid(events, self.grid_size)  # Shape: [2, H, W]

        # Determine corresponding image based on timestamps
        # Load exposure timestamps
        with open(timestamps_file, 'r') as f:
            lines = f.readlines()

        # Skip header lines and empty lines
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

        # Each line has: exposure_start_timestamp_us, exposure_end_timestamp_us
        # For each chunk, associate it with the image where chunk's end_time <= exposure_end_timestamp_us

        # Find the image index where the chunk's end_time <= exposure_end_timestamp_us
        # For simplicity, associate each chunk with the first image where exposure_end_timestamp_us >= last event timestamp
        last_event_ts = t[-1]

        image_idx = None
        for i, line in enumerate(lines):
            parts = line.split(',')
            if len(parts) >= 2:
                exposure_end_ts = int(parts[1].strip())
                if last_event_ts <= exposure_end_ts:
                    image_idx = i
                    break

        if image_idx is None:
            # If not found, use the last image
            image_idx = len(lines) - 1

        # Construct image paths
        # Correctly determine the sequence directory by going up three levels from events_file
        # Example: /content/Dataset/interlaken_00_a/events/left/events.h5 -> /content/Dataset/interlaken_00_a/
        seq_dir = os.path.dirname(os.path.dirname(os.path.dirname(events_file)))
        image_path = os.path.join(seq_dir, 'images', self.stream, 'rectified', f"{image_idx:06d}.png")
        image_prime_idx = min(image_idx + 1, len(lines) - 1)
        image_prime_path = os.path.join(seq_dir, 'images', self.stream, 'rectified', f"{image_prime_idx:06d}.png")

        # Load images
        image = self._load_image(image_path)
        image_prime = self._load_image(image_prime_path)

        sample = {
            'event_voxel_grid': voxel_grid,    # (2, H, W)
            'I_k': image,                       # (C, H, W)
            'I_k_prime': image_prime            # (C, H, W)
        }

        return sample

    def _load_image(self, image_file):
        """
        Load an image file and apply transformations.

        Args:
            image_file (str): Path to the image file.

        Returns:
            image (torch.Tensor): Transformed image tensor of shape [C, H, W].
        """
        if not os.path.exists(image_file):
            print(f"Image file not found: {image_file}")
            # Handle missing image, e.g., return a zero tensor or duplicate the last available image
            return torch.zeros(3, self.grid_size[0], self.grid_size[1])  # Assuming 3 channels for color

        image = Image.open(image_file).convert('RGB')  # Convert to RGB for color processing
        if self.transform:
            image = self.transform(image)  # Shape: (C, H, W)
        else:
            # Default transformation
            transform_default = transforms.ToTensor()
            image = transform_default(image)
        return image

    def _augment_data(self, voxel_grid, I_k, I_k_prime):
        """
        Applies random horizontal flipping and intensity jitter to the voxel grid and images.

        Args:
            voxel_grid (torch.Tensor): Tensor of shape [2, H, W]
            I_k (torch.Tensor): Tensor of shape [C, H, W]
            I_k_prime (torch.Tensor): Tensor of shape [C, H, W]

        Returns:
            tuple: Augmented voxel_grid, I_k, I_k_prime
        """
        # Random horizontal flip
        if np.random.rand() < 0.5:
            voxel_grid = torch.flip(voxel_grid, dims=[-1])    # Flip width dimension
            I_k = torch.flip(I_k, dims=[-1])
            I_k_prime = torch.flip(I_k_prime, dims=[-1])

        # Intensity jitter (brightness, contrast, saturation, hue for color images)
        if np.random.rand() < 0.5 and I_k.size(0) == 3:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            saturation = np.random.uniform(0.8, 1.2)
            hue = np.random.uniform(-0.05, 0.05)

            I_k = I_k * brightness
            I_k = (I_k - I_k.mean()) * contrast + I_k.mean()
            I_k = I_k.clamp(0.0, 1.0)

            I_k_prime = I_k_prime * brightness
            I_k_prime = (I_k_prime - I_k_prime.mean()) * contrast + I_k_prime.mean()
            I_k_prime = I_k_prime.clamp(0.0, 1.0)

        return voxel_grid, I_k, I_k_prime
