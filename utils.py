import copy
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch_geometric.nn import global_add_pool


def apply_radial_basis(predictions, vertices, inlet_vectors, epsilon=1e-8):
    force_norm = inlet_vectors.norm(p=2, dim=1, keepdim=True)
    e1 = inlet_vectors / (force_norm + epsilon)

    centroid = torch.mean(vertices, dim=0, keepdim=True)
    radial_vector = vertices - centroid
    r_unit = F.normalize(radial_vector, p=2, dim=1)

    e2_raw = torch.cross(e1, r_unit, dim=1)
    e2_norm = e2_raw.norm(p=2, dim=1, keepdim=True)
    e2 = e2_raw / (e2_norm + epsilon)

    e3 = torch.cross(e1, e2, dim=1)

    global_deflection = (
        (predictions[:, 0:1] * e1)
        + (predictions[:, 1:2] * e2)
        + (predictions[:, 2:3] * e3)
    )
    return global_deflection


def sample_uniform_so3(device='cpu', dtype=torch.float32):
    """Haar-uniform SO(3) sampling via random quaternion."""
    u1, u2, u3 = torch.rand(3, device=device, dtype=dtype)

    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)

    x, y, z, w = q1, q2, q3, q4

    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ]).to(device=device, dtype=dtype)


def _rotation_matrix_from_axis_angle(axis, theta, device='cpu', dtype=torch.float32):
    c = math.cos(theta)
    s = math.sin(theta)
    if axis == 'x':
        matrix = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'y':
        matrix = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        matrix = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return torch.tensor(matrix, device=device, dtype=dtype)


def get_rotation_matrix(mode, device='cpu'):
    if mode == 'canonical':
        return torch.eye(3, device=device, dtype=torch.float32)

    axis = np.random.choice(['x', 'y', 'z'])

    if mode == 'discrete':
        theta = float(np.radians(np.random.choice([0, 90, 180, 270])))
        return _rotation_matrix_from_axis_angle(axis, theta, device=device)
    elif mode == 'arbitrary':
        return sample_uniform_so3(device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'discrete', 'arbitrary', or 'canonical'.")


def rotate_tensor_field(field, rotation_matrix):
    if field is None:
        return None
    if field.ndim >= 2 and field.shape[-1] == 3:
        return torch.matmul(field, rotation_matrix.T)
    return field


def rotate_graph_data(data, rotation_matrix, args, rotate_targets=True):
    data = copy.deepcopy(data)

    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x - data.x.mean(dim=0, keepdim=True)
        data.x = torch.matmul(data.x, rotation_matrix.T)

    if hasattr(data, 'inlet_vel_direction') and data.inlet_vel_direction is not None:
        data.inlet_vel_direction = torch.matmul(data.inlet_vel_direction, rotation_matrix.T)

    if rotate_targets:
        if hasattr(data, 'y_wallShearStress') and data.y_wallShearStress is not None:
            data.y_wallShearStress = torch.matmul(data.y_wallShearStress, rotation_matrix.T)
        if hasattr(data, 'y') and data.y is not None:
            data.y = rotate_tensor_field(data.y, rotation_matrix)

    return data


def compute_metrics(pred, targets, batch_index=None):
    """Calculates MSE, MAE, RMSE, Max AE, Rel L2, Rel L1, R2."""
    diff = pred - targets
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(mse)
    max_ae = torch.max(torch.abs(diff))

    if batch_index is not None and len(pred.shape) > 1:
        err_sq_sum = global_add_pool(diff ** 2, batch_index)
        err_norm = torch.sqrt(err_sq_sum)
        target_sq_sum = global_add_pool(targets ** 2, batch_index)
        target_norm = torch.sqrt(target_sq_sum)
        rel_l2 = torch.mean(err_norm / (target_norm + 1e-8))

        err_abs_sum = global_add_pool(torch.abs(diff), batch_index)
        target_abs_sum = global_add_pool(torch.abs(targets), batch_index)
        rel_l1 = torch.mean(err_abs_sum / (target_abs_sum + 1e-8))
    else:
        err_norm = torch.norm(diff, p=2, dim=1)
        target_norm = torch.norm(targets, p=2, dim=1)
        rel_l2 = torch.mean(err_norm / (target_norm + 1e-8))

        err_norm_l1 = torch.norm(diff, p=1, dim=1)
        target_norm_l1 = torch.norm(targets, p=1, dim=1)
        rel_l1 = torch.mean(err_norm_l1 / (target_norm_l1 + 1e-8))

    try:
        r2 = r2_score(targets.detach().cpu().numpy(), pred.detach().cpu().numpy())
    except Exception:
        r2 = 0.0

    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse,
        'Max_AE': max_ae, 'Rel_L2': rel_l2, 'Rel_L1': rel_l1, 'R2': r2
    }


def relative_equivariance_error(pred_reference, pred_transformed, rotation_matrix):
    expected = rotate_tensor_field(pred_reference, rotation_matrix)
    numerator = torch.norm(pred_transformed - expected)
    denominator = torch.norm(expected) + 1e-8
    return numerator / denominator
