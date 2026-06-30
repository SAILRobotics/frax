import numpy as np

from frax.core.manipulator import Manipulator

_DEFAULT_URDF = None  # set at call time


def load_ur10e(
    urdf_filename: str,
    ee_offset: np.ndarray | None = None,
    collision_data: dict | None = None,
) -> Manipulator:
    """Create a frax Manipulator for the UR10e.

    Args:
        urdf_filename: Absolute path to ur10e.urdf.
        ee_offset: Optional 4x4 TCP offset from the last joint frame. Identity if None.
        collision_data: Optional frax-format collision dict (positions, radii,
            root_positions, root_radii, root_sc_pairs, root_sc_tols,
            body_sc_pairs, body_sc_tols). See oscbf/core/ur_collision_model.py.

    Returns:
        Manipulator with kinematics/dynamics expressed in the robot's base frame.
    """
    if ee_offset is None:
        ee_offset = np.eye(4)
    return Manipulator(
        urdf_filename=urdf_filename,
        ee_offset=ee_offset,
        collision_data=collision_data,
    )
