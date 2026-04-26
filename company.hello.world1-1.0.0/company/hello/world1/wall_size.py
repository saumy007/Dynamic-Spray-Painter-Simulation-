"""Pure math helper — no omni / warp dependencies."""
import math


def get_wall_size_xy_from_matrix(matrix, size):
    scale_x = math.sqrt(matrix[0][0]**2 + matrix[1][0]**2 + matrix[2][0]**2)
    scale_y = math.sqrt(matrix[0][1]**2 + matrix[1][1]**2 + matrix[2][1]**2)
    scale_z = math.sqrt(matrix[0][2]**2 + matrix[1][2]**2 + matrix[2][2]**2)
    s = float(size)
    return float(scale_x)*s, float(scale_y)*s, float(scale_z)*s


def get_plane_size_from_matrix(matrix, width_attr, length_attr):
    """
    UsdGeom.Plane lies in local XY; normal = local +Z.
    world_width  = scale_x * width_attr   (local X on face)
    world_height = scale_y * length_attr  (local Y on face)
    """
    scale_x = math.sqrt(matrix[0][0]**2 + matrix[1][0]**2 + matrix[2][0]**2)
    scale_y = math.sqrt(matrix[0][1]**2 + matrix[1][1]**2 + matrix[2][1]**2)
    return float(scale_x) * float(width_attr), float(scale_y) * float(length_attr)
