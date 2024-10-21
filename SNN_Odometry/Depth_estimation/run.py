# run.py with validation

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
#from model import SNNDepthPoseEstimator  # Ensure model.py is correctly implemented and updated
#from loss_functions import cross_modal_consistency_loss, compute_depth_metrics  # Import new loss functions and metrics
#from loss_functions import reprojection_loss  # If needed elsewhere
import snntorch as snn
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
from torchinfo import summary  # Use torchinfo instead of torchsummary
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import time
import gc
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# Set CUDA Allocator Configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# loss_functions.py

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
# loss_functions.py

from pytorch_msssim import ssim
import torch.nn.functional as F

def reprojection_loss(I_k, I_k_reconstructed, alpha=0.85):
    """
    Computes the reprojection loss between the original and reconstructed images.

    Args:
        I_k (torch.Tensor): Original intensity image [B, C, H, W]
        I_k_reconstructed (torch.Tensor): Reconstructed intensity image [B, C, H, W]
        alpha (float): Weight for SSIM

    Returns:
        torch.Tensor: Combined loss
    """
    # Ensure inputs are float32
    I_k = I_k.float()
    I_k_reconstructed = I_k_reconstructed.float()

    # Debug: Print tensor dtypes
    print(f"[reprojection_loss] I_k dtype: {I_k.dtype}, I_k_reconstructed dtype: {I_k_reconstructed.dtype}")

    # Compute SSIM
    ssim_val = ssim(I_k, I_k_reconstructed, data_range=1.0, size_average=True)

    # Compute L1 Loss
    l1_loss = F.l1_loss(I_k, I_k_reconstructed, reduction='mean')

    # Combine losses
    loss = alpha * (1 - ssim_val) + (1 - alpha) * l1_loss

    return loss
# loss_functions.py

from pytorch_msssim import ssim
import torch.nn.functional as F

def compute_image_metrics(reconstructed, target):
    """
    Computes image-based metrics such as SSIM and L1 loss.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor [B, C, H, W]
        target (torch.Tensor): Target image tensor [B, C, H, W]

    Returns://
        dict: Dictionary containing image-based metrics.
    """
    # Ensure tensors are float32
    if reconstructed.dtype != torch.float32:
        reconstructed = reconstructed.float()
    if target.dtype != torch.float32:
        target = target.float()

    # Debug: Print tensor dtypes
    print(f"[compute_image_metrics] reconstructed dtype: {reconstructed.dtype}, target dtype: {target.dtype}")

    metrics = {}
    try:
        ssim_val = ssim(reconstructed, target, data_range=1.0, size_average=True).item()
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        ssim_val = 0.0  # Assign a default value or handle as needed
    l1_loss = F.l1_loss(reconstructed, target, reduction='mean').item()
    metrics['SSIM'] = ssim_val
    metrics['L1'] = l1_loss
    return metrics

def inverse_warp(I_k_prime, depth, pose, intrinsic, rotation_mode='euler', padding_mode='zeros'):
    """
    Reconstruct the intensity frame using depth and pose.
    """
    B, C, H, W = I_k_prime.size()
    device = I_k_prime.device

    # Create mesh grid
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    grid_x = grid_x.float().view(1, -1).repeat(B, 1)  # [B, H*W]
    grid_y = grid_y.float().view(1, -1).repeat(B, 1)  # [B, H*W]

    # Stack to get pixel coordinates
    pix_coords = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=1)  # [B, 3, H*W]

    # Invert intrinsic matrix
    intrinsic_inv = torch.inverse(intrinsic)  # [B, 3, 3]
    cam_coords = torch.bmm(intrinsic_inv, pix_coords)  # [B, 3, H*W]

    # Multiply by depth
    depth_flat = depth.view(B, 1, -1)  # [B,1,H*W]
    cam_coords = cam_coords * depth_flat  # [B, 3, H*W]

    # Convert to homogeneous coordinates
    cam_coords_hom = torch.cat([cam_coords, torch.ones((B, 1, H*W), device=device)], dim=1)  # [B,4,H*W]

    # Convert pose from [B,6] to [B,3,4]
    if rotation_mode == 'euler':
        rotation = pose[:, :3]  # [B,3]
        translation = pose[:, 3:]  # [B,3]
        rotation_matrix = euler_angles_to_rotation_matrix(rotation)  # [B,3,3]
    elif rotation_mode == 'matrix':
        rotation_matrix = pose[:, :9].view(B, 3, 3)  # [B,3,3]
        translation = pose[:, 9:].view(B, 3, 1)  # [B,3,1]
    else:
        raise ValueError("Unsupported rotation mode. Choose 'euler' or 'matrix'.")

    translation = translation.view(B, 3, 1)  # [B,3,1]

    # Combine rotation and translation to form pose matrix [B,3,4]
    pose_matrix = torch.cat([rotation_matrix, translation], dim=2)  # [B,3,4]

    # Apply pose transformation
    cam_transformed = torch.bmm(pose_matrix, cam_coords_hom)  # [B,3,H*W]

    # Perspective division
    cam_transformed = cam_transformed / (cam_transformed[:, 2:3, :] + 1e-6)  # [B,3,H*W]

    # Project back to pixel coordinates
    I_k_reconstructed = torch.bmm(intrinsic, cam_transformed)  # [B,3,H*W]
    I_k_reconstructed = I_k_reconstructed.view(B, 3, H, W)  # [B,3,H,W]

    return I_k_reconstructed


# Define spike encoding functions (Ensure these are correctly implemented)
def encode_voxel_grid(voxel_grid, time_steps):
    """
    Encode voxel grid into spikes over time steps using rate coding.

    Args:
        voxel_grid (torch.Tensor): Tensor of shape [B, C, H, W]
        time_steps (int): Number of time steps for spike encoding

    Returns:
        torch.Tensor: Spikes tensor of shape [T, B, C, H, W]
    """
    # Normalize voxel grid to [0, 1]
    voxel_grid_normalized = voxel_grid / (voxel_grid.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-6)
    # Clamp to ensure values are within [0, 1]
    voxel_grid_normalized = torch.clamp(voxel_grid_normalized, 0.0, 1.0)
    # Replace NaNs or Infs with zeros
    voxel_grid_normalized = torch.nan_to_num(voxel_grid_normalized)
    # Repeat for each time step
    voxel_grid_repeated = voxel_grid_normalized.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)  # (T, B, C, H, W)
    # Generate spikes using Bernoulli encoding
    spikes = torch.bernoulli(voxel_grid_repeated)
    return spikes  # Shape: (T, B, C, H, W)

def encode_intensity_frame(intensity_frame, time_steps):
    """
    Encode intensity frames into spikes over time steps using rate coding.

    Args:
        intensity_frame (torch.Tensor): Tensor of shape [B, C, H, W]
        time_steps (int): Number of time steps for spike encoding

    Returns:
        torch.Tensor: Spikes tensor of shape [T, B, C, H, W]
    """
    # Normalize intensity frame to [0, 1]
    intensity_normalized = intensity_frame / (intensity_frame.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-6)
    # Clamp to ensure values are within [0, 1]
    intensity_normalized = torch.clamp(intensity_normalized, 0.0, 1.0)
    # Replace NaNs or Infs with zeros
    intensity_normalized = torch.nan_to_num(intensity_normalized)
    # Repeat for each time step
    intensity_repeated = intensity_normalized.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)  # (T, B, C, H, W)
    # Generate spikes using Bernoulli encoding
    spikes = torch.bernoulli(intensity_repeated)
    return spikes  # Shape: (T, B, C, H, W)

# Define a helper function to convert Euler angles to rotation matrices (Ensure correctness)
def euler_angles_to_rotation_matrix(theta):
    """
    Convert Euler angles to rotation matrices.
    """
    B = theta.size(0)
    R_x = torch.zeros(B, 3, 3, device=theta.device)
    R_y = torch.zeros(B, 3, 3, device=theta.device)
    R_z = torch.zeros(B, 3, 3, device=theta.device)

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # Rotation around x-axis
    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos[:, 0]
    R_x[:, 1, 2] = -sin[:, 0]
    R_x[:, 2, 1] = sin[:, 0]
    R_x[:, 2, 2] = cos[:, 0]

    # Rotation around y-axis
    R_y[:, 0, 0] = cos[:, 1]
    R_y[:, 0, 2] = sin[:, 1]
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin[:, 1]
    R_y[:, 2, 2] = cos[:, 1]

    # Rotation around z-axis
    R_z[:, 0, 0] = cos[:, 2]
    R_z[:, 0, 1] = -sin[:, 2]
    R_z[:, 1, 0] = sin[:, 2]
    R_z[:, 1, 1] = cos[:, 2]
    R_z[:, 2, 2] = 1

    # Combined rotation matrix
    R = torch.bmm(R_z, torch.bmm(R_y, R_x))  # [B, 3, 3]

    return R
'''
# Define a helper function to compute image-based metrics
def compute_image_metrics(reconstructed, target):
    """
    Computes image-based metrics such as SSIM and L1 loss.

    Args:
        reconstructed (torch.Tensor): Reconstructed image. Shape: [B, C, H, W]
        target (torch.Tensor): Target image. Shape: [B, C, H, W]

    Returns:
        dict: Dictionary containing image-based metrics.
    """
    metrics = {}
    ssim_val = ssim(reconstructed, target, data_range=1.0, size_average=True).item()
    l1_loss = F.l1_loss(reconstructed, target, reduction='mean').item()
    metrics['SSIM'] = ssim_val
    metrics['L1'] = l1_loss
    return metrics

from pytorch_msssim import ssim  # Ensure ssim is imported
'''
def compute_image_metrics(reconstructed, target):
    """
    Computes image-based metrics such as SSIM and L1 loss.

    Args:
        reconstructed (torch.Tensor): Reconstructed image tensor [B, C, H, W]
        target (torch.Tensor): Target image tensor [B, C, H, W]

    Returns:
        dict: Dictionary containing image-based metrics.
    """
    # Ensure tensors are float32
    if reconstructed.dtype != torch.float32:
        reconstructed = reconstructed.float()
    if target.dtype != torch.float32:
        target = target.float()

    metrics = {}
    try:
        ssim_val = ssim(reconstructed, target, data_range=1.0, size_average=True).item()
    except Exception as e:
        print(f"Error computing SSIM: {e}")
        ssim_val = 0.0  # Assign a default value or handle as needed
    l1_loss = F.l1_loss(reconstructed, target, reduction='mean').item()
    metrics['SSIM'] = ssim_val
    metrics['L1'] = l1_loss
    return metrics

# Define a helper function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints'):
    """
    Saves the model checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Initialize Hyperparameters
num_epochs = 10
learning_rate = 1e-4
batch_size = 8
num_workers = 6
chunk_size = 1000000
grid_size = (320, 640)
time_steps = 10
subset_files = 7#None
accumulation_steps = 2  # For gradient accumulation (optional)

# Define image transformations as per the paper
image_transforms = transforms.Compose([
    transforms.Resize((480, 640)),  # Resize to match grid_size before cropping
    transforms.CenterCrop(grid_size),  # Center crop to 320x640
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0.0, 1.0]
])

# Set the root directory of the DSEC dataset
root_dir = '/content/drive/MyDrive/MTech_capstone/Depth_estimation/Dataset'  # Replace with your actual path
checkpoint_dir='/content/drive/MyDrive/MTech_capstone/Depth_estimation/checkpoints'

# Select stream ('left' or 'right')
stream = 'left'  # Change to 'right' if needed

# Create the dataset instances
train_dataset = DSECDataset(
    directory_path=root_dir,
    chunk_size=chunk_size,
    grid_size=grid_size,
    stream=stream,
    transform=image_transforms,
    mode='train'  # Specify mode for training
)

validation_dataset = DSECDataset(
    directory_path=root_dir,
    chunk_size=chunk_size,
    grid_size=grid_size,
    stream=stream,
    transform=image_transforms,
    mode='validation'  # Specify mode for validation
)

# Create DataLoader for training
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # Shuffle for training
    num_workers=num_workers,
    pin_memory=True,  # Enable if using CUDA
    prefetch_factor=2
)

# Create DataLoader for validation
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle for validation
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2
)

# After creating the DataLoader
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(validation_loader)}")

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# I added this

# Function to print tensor shapes and statistics
def print_sample_info(dataloader, num_batches=1, num_samples=3):
    """
    Print shapes and statistics of samples from the dataloader.
    """
    for batch_idx, sample in enumerate(dataloader):
        if batch_idx >= num_batches:
            break  # Only inspect a limited number of batches

        voxel_grids = sample['event_voxel_grid']    # (B, 2, H, W)
        I_k = sample['I_k']                         # (B, C, H, W)
        I_k_prime = sample['I_k_prime']             # (B, C, H, W)

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Voxel Grid Shape: {voxel_grids.shape}")
        print(f"  I_k Shape: {I_k.shape}")
        print(f"  I_k_prime Shape: {I_k_prime.shape}")

        # Print min and max values
        print(f"  Voxel Grid - Min: {voxel_grids.min().item()}, Max: {voxel_grids.max().item()}")
        print(f"  I_k - Min: {I_k.min().item()}, Max: {I_k.max().item()}")
        print(f"  I_k_prime - Min: {I_k_prime.min().item()}, Max: {I_k_prime.max().item()}")

        # Optionally, print mean and standard deviation
        print(f"  Voxel Grid - Mean: {voxel_grids.mean().item():.4f}, Std: {voxel_grids.std().item():.4f}")
        print(f"  I_k - Mean: {I_k.mean().item():.4f}, Std: {I_k.std().item():.4f}")
        print(f"  I_k_prime - Mean: {I_k_prime.mean().item():.4f}, Std: {I_k_prime.std().item():.4f}")

        # Check for NaNs or Infs
        if torch.isnan(voxel_grids).any() or torch.isinf(voxel_grids).any():
            print("  Voxel Grid contains NaNs or Infs!")
        if torch.isnan(I_k).any() or torch.isinf(I_k).any():
            print("  I_k contains NaNs or Infs!")
        if torch.isnan(I_k_prime).any() or torch.isinf(I_k_prime).any():
            print("  I_k_prime contains NaNs or Infs!")

# Call the function to print sample info
print_sample_info(train_loader, num_batches=1, num_samples=3)

def visualize_samples(dataloader, num_batches=1, num_samples=3, resize=False, new_size=(320, 640)):
    """
    Visualize samples from the dataloader.
    """
    for batch_idx, sample in enumerate(dataloader):
        if batch_idx >= num_batches:
            break  # Only visualize a limited number of batches

        voxel_grids = sample['event_voxel_grid']    # (B, 2, H, W)
        I_k = sample['I_k']                         # (B, C, H, W)
        I_k_prime = sample['I_k_prime']             # (B, C, H, W)

        batch_size = voxel_grids.size(0)

        for i in range(min(num_samples, batch_size)):
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))

            # Voxel Grid - Polarity
            polarity_grid = voxel_grids[i, 0].cpu().numpy()
            if resize:
                polarity_grid = cv2.resize(polarity_grid, new_size, interpolation=cv2.INTER_AREA)
            axs[0].imshow(polarity_grid, cmap='gray')
            axs[0].set_title('Event Polarity Grid')
            axs[0].axis('off')

            # Voxel Grid - Event Count
            count_grid = voxel_grids[i, 1].cpu().numpy()
            if resize:
                count_grid = cv2.resize(count_grid, new_size, interpolation=cv2.INTER_AREA)
            axs[1].imshow(count_grid, cmap='viridis')
            axs[1].set_title('Event Count Grid')
            axs[1].axis('off')

            # Image I_k
            img_I_k = I_k[i].permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
            if resize:
                img_I_k = cv2.resize(img_I_k, new_size, interpolation=cv2.INTER_AREA)
            axs[2].imshow(img_I_k)  # Display RGB image
            axs[2].set_title('Image I_k')
            axs[2].axis('off')

            # Image I_k_prime
            img_I_k_prime = I_k_prime[i].permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
            if resize:
                img_I_k_prime = cv2.resize(img_I_k_prime, new_size, interpolation=cv2.INTER_AREA)
            axs[3].imshow(img_I_k_prime)  # Display RGB image
            axs[3].set_title('Image I_k_prime')
            axs[3].axis('off')

            plt.tight_layout()
            plt.show()

# Example: Visualize first 3 samples from the first batch
visualize_samples(train_loader, num_batches=1, num_samples=3, resize=True, new_size=(320, 640))


try:
    torch.cuda.empty_cache()
    model = SNNDepthPoseEstimator().to(device)
    print("Model initialized successfully.")

    # Print model summary using torchinfo
    summary(
        model,
        input_data=[
            torch.randn(batch_size, 2, grid_size[0], grid_size[1]).to(device),    # event_voxel_grid
            torch.randn(batch_size, 3, grid_size[0], grid_size[1]).to(device),    # I_k
            torch.randn(batch_size, 3, grid_size[0], grid_size[1]).to(device)     # I_k_prime
        ],
        depth=3
    )
except Exception as e:
    print(f"Error during model initialization: {e}")
    raise

# Initialize optimizer and GradScaler
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


scaler = GradScaler()  # Defaults to CPU


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='/content/drive/MyDrive/MTech_capstone/Depth_estimation/logs')

# Define intrinsic matrix (replace with actual calibration)
def get_intrinsic_matrix(batch_size, device):
    """
    Returns the intrinsic matrix replicated for the batch size.

    Args:
        batch_size (int): Number of samples in the batch.
        device (torch.device): Device to place the intrinsic matrix.

    Returns:
        torch.Tensor: Intrinsic matrices of shape [B, 3, 3]
    """
    fx, fy = 600.0, 600.0
    cx, cy = 320.0, 240.0
    intrinsic = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=torch.float32).to(device)
    intrinsic = intrinsic.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
    return intrinsic

# Define a backward wrapper function
def backward_wrapper(loss, scaler, retain_graph=False):
    """
    Wrapper function for backward pass using GradScaler.
    """
    scaler.scale(loss).backward(retain_graph=retain_graph)

# Training and Validation Loop
global_step = 0
for epoch in range(1, num_epochs + 1):
    # Training Phase
    model.train()
    epoch_loss = 0.0
    epoch_start_time = time.time()

    for batch_idx, sample in enumerate(train_loader, 1):
        voxel_grid = sample['event_voxel_grid'].to(device)    # [B, 2, H, W]
        I_k = sample['I_k'].to(device)                       # [B, C, H, W]
        I_k_prime = sample['I_k_prime'].to(device)           # [B, C, H, W]
        # gt_depth = sample['gt_depth'].to(device)           # Removed since gt_depth is unavailable
        batch_size_current = voxel_grid.size(0)

        # Reset neuron states before the forward pass
        model.reset_neurons()

        # Update intrinsic matrix to match the current batch size
        intrinsic_batch = get_intrinsic_matrix(batch_size_current, device)

        # Encode inputs into spikes
        event_spikes = encode_voxel_grid(voxel_grid, time_steps).to(device)          # [T, B, C, H, W]
        I_k_spikes = encode_intensity_frame(I_k, time_steps).to(device)              # [T, B, C, H, W]
        I_k_prime_spikes = encode_intensity_frame(I_k_prime, time_steps).to(device)  # [T, B, C, H, W]

        # Initialize lists to collect outputs over time
        disparity_list = []
        pose_list = []

        # Zero the gradients before the forward pass
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):  # Enable mixed precision
            for t in range(time_steps):
                # Extract spikes at time step t
                event_input = event_spikes[t]         # [B, C, H, W]
                image_input = I_k_spikes[t]           # [B, C, H, W]
                image_prime_input = I_k_prime_spikes[t]  # [B, C, H, W]

                # Forward pass
                disparity, pose = model(event_input, image_input, image_prime_input)  # disparity: [B, 1, H, W], pose: [B, 6]
                disparity_list.append(disparity)
                pose_list.append(pose)

            # Use the last time step's output for loss computation
            disparity_pred = disparity_list[-1]  # [B, 1, H, W]
            pose_pred = pose_list[-1]            # [B, 6]

            # Compute depth from disparity
            depth_pred = 1.0 / (disparity_pred + 1e-6)  # Avoid division by zero

            # Upsample depth_pred to match I_k_prime's size (320x640)
            depth_pred_upsampled = F.interpolate(
                depth_pred,
                size=(320, 640),
                mode='bilinear',
                align_corners=False
            )  # [B, 1, 320, 640]

            # Compute reconstructed intensity frame using the upsampled depth and pose
            I_k_reconstructed = inverse_warp(
                I_k_prime,
                depth_pred_upsampled,  # [B, 1, 320, 640]
                pose_pred,             # [B, 6]
                intrinsic_batch,
                rotation_mode='euler',  # Or 'quaternion' based on your implementation
                padding_mode='zeros'
            )  # [B, C, 320, 640]

        # Compute Reprojection Loss between I_k and I_k_reconstructed
        loss = reprojection_loss(I_k, I_k_reconstructed, alpha=0.85)

        # Backward pass and optimization
        backward_wrapper(loss, scaler)  # Backpropagate on the current batch's loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        epoch_loss += loss.item()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), global_step)
        global_step += 1

        # Optional: Visualize samples periodically (e.g., first batch of first epoch)
        if epoch == 1 and batch_idx == 1:
            # Uncomment the following line if you want to visualize during training
            # visualize_samples(train_loader, num_batches=1, num_samples=3, resize=True, new_size=(320, 640))
            pass

        # Print batch summary (remove debug statements)
        print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Compute average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    # Print epoch summary
    print(f"Epoch [{epoch}/{num_epochs}] completed in {epoch_duration:.2f} seconds | Average Loss: {avg_epoch_loss:.4f}")

    # Save the model checkpoint
    save_checkpoint(model, optimizer, epoch, checkpoint_dir=checkpoint_dir)

    # Validation Phase
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        all_metrics = {
            'SSIM': 0.0,
            'L1': 0.0
        }
        num_val_batches = len(validation_loader)
        if num_val_batches == 0:
            print("Warning: Validation loader has 0 batches. Skipping validation.")
        else:
            for val_batch_idx, val_sample in enumerate(validation_loader, 1):
                voxel_grid_val = val_sample['event_voxel_grid'].to(device)    # [B, 2, H, W]
                I_k_val = val_sample['I_k'].to(device)                       # [B, C, H, W]
                I_k_prime_val = val_sample['I_k_prime'].to(device)           # [B, C, H, W]
                # gt_depth_val = val_sample['gt_depth'].to(device)           # Removed since gt_depth is unavailable
                batch_size_val = voxel_grid_val.size(0)

                # Reset neuron states
                model.reset_neurons()

                # Update intrinsic matrix
                intrinsic_val = get_intrinsic_matrix(batch_size_val, device)

                # Encode inputs
                event_spikes_val = encode_voxel_grid(voxel_grid_val, time_steps).to(device)          # [T, B, C, H, W]
                I_k_spikes_val = encode_intensity_frame(I_k_val, time_steps).to(device)              # [T, B, C, H, W]
                I_k_prime_spikes_val = encode_intensity_frame(I_k_prime_val, time_steps).to(device)  # [T, B, C, H, W]

                # Initialize lists to collect outputs over time
                disparity_list_val = []
                pose_list_val = []

                with torch.amp.autocast(device_type='cuda'):
                    for t in range(time_steps):
                        # Extract spikes at time step t
                        event_input_val = event_spikes_val[t]         # [B, C, H, W]
                        image_input_val = I_k_spikes_val[t]           # [B, C, H, W]
                        image_prime_input_val = I_k_prime_spikes_val[t]  # [B, C, H, W]

                        # Forward pass
                        disparity_val, pose_val = model(event_input_val, image_input_val, image_prime_input_val)  # [B,1,H,W], [B,6]
                        disparity_list_val.append(disparity_val)
                        pose_list_val.append(pose_val)

                    # Use the last time step's output
                    disparity_pred_val = disparity_list_val[-1]  # [B,1,H,W]
                    pose_pred_val = pose_list_val[-1]            # [B,6]

                    # Compute depth
                    depth_pred_val = 1.0 / (disparity_pred_val + 1e-6)  # [B,1,H,W]

                    # Upsample depth
                    depth_pred_val_upsampled = F.interpolate(
                        depth_pred_val,
                        size=(320, 640),
                        mode='bilinear',
                        align_corners=False
                    )  # [B,1,320,640]

                    # Reconstruct intensity frame
                    I_k_reconstructed_val = inverse_warp(
                        I_k_prime_val,
                        depth_pred_val_upsampled,
                        pose_pred_val,
                        intrinsic_val,
                        rotation_mode='euler',  # Or 'quaternion'
                        padding_mode='zeros'
                    )  # [B, C, 320, 640]

                # Compute Reprojection Loss between I_k_val and I_k_reconstructed_val
                loss_val = reprojection_loss(I_k_val, I_k_reconstructed_val, alpha=0.85)
                val_loss += loss_val.item()

                # Compute image-based metrics (SSIM and L1)
                metrics = compute_image_metrics(I_k_reconstructed_val, I_k_val)
                for metric, value in metrics.items():
                    if metric in all_metrics:
                        all_metrics[metric] += value

            # Average validation loss and metrics
            avg_val_loss = val_loss / num_val_batches
            for metric in all_metrics:
                all_metrics[metric] /= num_val_batches

            # Log validation loss and metrics to TensorBoard
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            for metric, value in all_metrics.items():
                writer.add_scalar(f'Eval/{metric}', value, epoch)

            # Print validation summary
            print(f"Validation Epoch [{epoch}/{num_epochs}] | Loss: {avg_val_loss:.4f}")
            for metric, value in all_metrics.items():
                print(f"  {metric}: {value:.4f}")

    # Clear CUDA cache and collect garbage after each epoch
    torch.cuda.empty_cache()
    gc.collect()

# Close the TensorBoard writer after training
writer.close()
