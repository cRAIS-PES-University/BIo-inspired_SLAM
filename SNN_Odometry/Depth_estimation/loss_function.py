# loss_functions.py

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

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

def cross_modal_consistency_loss(I_k, I_k_reconstructed_kprime, I_k_reconstructed_self, alpha=0.85):
    """
    Computes the cross-modal consistency loss with auto-masking.

    Args:
        I_k (torch. Tensor): Original intensity frame. Shape: [B, C, H, W]
        I_k_reconstructed_kprime (dict): Reconstructed frames from adjacent frames {k': tensor}. Each tensor shape: [B, C, H, W]
        I_k_reconstructed_self (torch.Tensor): Reconstructed frame from itself. Shape: [B, C, H, W]
        alpha (float): Weighting factor for SSIM. Default is 0.85.

    Returns:
        torch.Tensor: Cross-modal consistency loss.
    """
    # Compute reprojection loss with adjacent frames
    losses = []
    pe_k_kprime = []
    for kprime, I_k_recon in I_k_reconstructed_kprime.items():
        loss = reprojection_loss(I_k, I_k_recon, alpha)
        losses.append(loss)
        pe_k_kprime.append(reprojection_loss(I_k, I_k_recon, alpha).detach())

    # Compute reprojection loss with itself
    pe_k_self = reprojection_loss(I_k, I_k_reconstructed_self, alpha).detach()

    # Stack reprojection errors
    pe_k_kprime = torch.stack(pe_k_kprime, dim=1)  # Shape: [B, num_kprime]
    pe_k_self = pe_k_self.unsqueeze(1)            # Shape: [B, 1]

    # Find the minimum reprojection error across k'
    min_pe_k_kprime, _ = pe_k_kprime.min(dim=1)   # Shape: [B]
    min_pe_k_self, _ = pe_k_self.min(dim=1)       # Shape: [B]

    # Auto-Masking
    mask = (min_pe_k_kprime < min_pe_k_self).float().unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Shape: [B,1,1,1]

    # Select the minimum loss
    min_loss = torch.stack(losses, dim=1).min(dim=1)[0]  # Shape: [B]

    # Apply mask
    masked_loss = min_loss * mask.squeeze()  # Shape: [B]

    # Average over batch
    final_loss = masked_loss.mean()
    return final_loss

def compute_depth_metrics(pred, gt, max_depths=[10, 20, 30]):
    """
    Computes depth evaluation metrics.

    Args:
        pred (torch.Tensor): Predicted depth. Shape: [B, 1, H, W]
        gt (torch.Tensor): Ground truth depth. Shape: [B, 1, H, W]
        max_depths (list): List of maximum depth cut-offs.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {}
    pred = pred.squeeze(1)  # [B, H, W]
    gt = gt.squeeze(1)      # [B, H, W]

    # Ensure no division by zero and clamp depth values
    pred = torch.clamp(pred, min=1e-3, max=100.0)
    gt = torch.clamp(gt, min=1e-3, max=100.0)

    # Compute absolute depth error
    abs_depth_error = torch.abs(pred - gt)

    # Compute Abs Rel
    metrics['Abs Rel'] = torch.mean(abs_depth_error / gt).item()

    # Compute Sq Rel
    metrics['Sq Rel'] = torch.mean((abs_depth_error ** 2) / gt).item()

    # Compute RMSE
    metrics['RMSE'] = torch.sqrt(torch.mean(abs_depth_error ** 2)).item()

    # Compute RMSE log
    metrics['RMSE log'] = torch.sqrt(torch.mean((torch.log(pred) - torch.log(gt)) ** 2)).item()

    # Compute SI log
    metrics['SI log'] = torch.mean(torch.abs(torch.log(pred) - torch.log(gt))).item()

    # Compute δ < 1.25, δ < 1.25^2, δ < 1.25^3
    max_delta = torch.max(gt / pred, pred / gt)
    metrics['δ<1.25'] = torch.mean((max_delta < 1.25).float()).item()
    metrics['δ<1.25^2'] = torch.mean((max_delta < 1.25 ** 2).float()).item()
    metrics['δ<1.25^3'] = torch.mean((max_delta < 1.25 ** 3).float()).item()

    # Compute average absolute depth errors at different max cut-offs
    for max_depth in max_depths:
        mask = gt <= max_depth
        avg_abs_error = torch.mean(abs_depth_error[mask]).item()
        metrics[f'Abs Error < {max_depth}m'] = avg_abs_error

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


def pose_vec2mat(pose, rotation_mode='euler'):
    """
    Convert 6DoF pose vector to transformation matrix.

    Args:
        pose (torch.Tensor): [B, 6] pose vectors.
        rotation_mode (str): 'euler' or 'quat'.

    Returns:
        T (torch.Tensor): [B, 4, 4] transformation matrices.
    """
    batch_size = pose.size(0)
    device = pose.device

    # Extract rotation and translation
    translation = pose[:, :3]  # [B, 3]
    rotation = pose[:, 3:]     # [B, 3]

    if rotation_mode == 'euler':
        rot_mat = euler_angles_to_rotation_matrix(rotation)
    elif rotation_mode == 'quat':
        rot_mat = quaternion_to_rotation_matrix(rotation)
    else:
        raise ValueError("rotation_mode should be 'euler' or 'quat'")

    # Construct transformation matrix
    T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 4, 4]
    T[:, :3, :3] = rot_mat
    T[:, :3, 3] = translation

    return T


def euler_angles_to_rotation_matrix(theta):
    """
    Convert Euler angles to rotation matrices.
    The angles are assumed to be in radians.
    Rotation order is XYZ.

    Args:
        theta (torch.Tensor): Tensor of shape [B, 3] representing Euler angles.

    Returns:
        torch.Tensor: Rotation matrices of shape [B, 3, 3].
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


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions (torch.Tensor): [B, 4] quaternions.

    Returns:
        rot_mat (torch.Tensor): [B, 3, 3] rotation matrices.
    """
    # Normalize quaternions
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    rot_mat = torch.stack([
        1 - 2*(y**2 + z**2),
        2*(x*y - z*w),
        2*(x*z + y*w),

        2*(x*y + z*w),
        1 - 2*(x**2 + z**2),
        2*(y*z - x*w),

        2*(x*z - y*w),
        2*(y*z + x*w),
        1 - 2*(x**2 + y**2)
    ], dim=1).reshape(-1, 3, 3)

    return rot_mat

def compute_loss(disparity_pred, disparity_gt=None, pose_pred=None, pose_gt=None, I_k=None, I_k_reconstructed=None):
    """
    Computes the combined loss for depth and pose estimations.

    Args:
        disparity_pred (torch.Tensor): Predicted disparity map [B, 1, H, W].
        disparity_gt (torch.Tensor, optional): Ground truth disparity map [B, 1, H, W].
        pose_pred (torch.Tensor, optional): Predicted pose [B, 6].
        pose_gt (torch.Tensor, optional): Ground truth pose [B, 6].
        I_k (torch.Tensor, optional): Original intensity frame [B, C, H, W].
        I_k_reconstructed (torch.Tensor, optional): Reconstructed intensity frame [B, C, H, W].

    Returns:
        loss (torch.Tensor): Combined loss value.
    """
    # Depth loss (self-supervised)
    # Since disparity_gt is not available, use reprojection loss
    if I_k is not None and I_k_reconstructed is not None:
        reproj_loss = reprojection_loss(I_k, I_k_reconstructed)
    else:
        reproj_loss = torch.tensor(0.0, device=disparity_pred.device)

    # Pose loss (optional, if ground truth poses are available)
    if pose_gt is not None and pose_pred is not None:
        pose_loss = F.mse_loss(pose_pred, pose_gt)
    else:
        pose_loss = torch.tensor(0.0, device=disparity_pred.device)

    # Optional: Add depth smoothness loss or other regularization
    # Example:
    # depth_smoothness = compute_depth_smoothness(disparity_pred, I_k)

    # Total loss
    loss = reproj_loss + pose_loss
    return loss
