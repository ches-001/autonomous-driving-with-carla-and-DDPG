# Some of the code in this file were inspired from:
# https://github.com/alberto-mate/CARLA-SB3-RL-Training-Environment/blob/main/carla_env/wrappers.py

import carla 
import numpy as np
from typing import Union


def get_actor_display_name(actor: carla.Actor, truncate: int=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def to_vector(v: Union[carla.Location, carla.Vector3D, carla.Rotation]) -> np.ndarray:
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])
    else:
         raise TypeError(f"unsupported type to vector {type(v)}")
    

def radian_angle_diff(v0: np.ndarray, v1: np.ndarray, threshold: float=2.3) -> float:
    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0
    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    dot_product = np.dot(v0_xy_u, v1_xy_u)
    angle = np.arccos(dot_product)
    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    # threshold (2.3 rad) is a big angle and would heavily penalize the RL agent
    # hence angles beyond this are filtered
    if abs(angle) >= threshold: return 0
    return angle


def smoothen_action(
        prev_value: Union[float, np.ndarray], 
        curent_value: Union[float, np.ndarray], 
        smooth_factor: Union[float, np.ndarray]
    ) -> np.ndarray:
    return (prev_value * smooth_factor) + (curent_value * (1 - smooth_factor))


def distance_to_line(x1: np.ndarray, x2: np.ndarray, p: np.ndarray) -> float:
        # compute the perpendicular distance between a line (passing through x1 and x2) and p
        p[2] = 0
        num = np.linalg.norm(np.cross(x2 - x1, x1 - p))
        denom = np.linalg.norm(x2 - x1)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - x1)
        return num / denom