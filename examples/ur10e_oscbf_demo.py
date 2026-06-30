"""UR10e OSCBF velocity-control demo using the frax package.

Kinematics & dynamics: frax.core.manipulator.Manipulator (JAX)
Collision bodies:      oscbf/core/ur_collision_model.py (sphere model)
Safety filter:         cbf_utils.OSCBFVelocityConfig (cbfpy)
Simulation & viewer:   MuJoCo 3.x with menagerie OBJ meshes

Run from the ProxyGrip root:
    conda activate proxy
    python main_logic/frax/examples/ur10e_oscbf_demo.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import mujoco
import mujoco.viewer
from cbfpy import CBF

from scipy.spatial.transform import Rotation as ScipyR

import frax
from frax.robots.ur10e import load_ur10e
from frax.utils.rotation_utils import orientation_error_3D

sys.path.insert(0, str(Path(__file__).parent))
from cbf_utils import OSCBFVelocityConfig

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def _find_proxygrip_root() -> Path:
    here = Path(__file__).absolute()
    for parent in (here, *here.parents):
        if (parent / "ur10e_bundle").is_dir():
            return parent
    raise FileNotFoundError("Cannot find ProxyGrip root (no ur10e_bundle/ directory found)")

_ROOT  = _find_proxygrip_root()
_URDF  = str(_ROOT / "ur10e_bundle" / "ur10e.urdf")
_PHASE1 = _ROOT / "main_logic" / "initialization" / "hand_eye_data" / "results" / "phase1_results.npz"


def _load_base_transform():
    """Return (base_pos, base_R): robot base pose in frax world frame.

    Loaded from phase1_results.npz when available; falls back to identity.
    """
    if _PHASE1.exists():
        T = np.load(_PHASE1)["T_world_base"]
        return T[:3, 3].copy(), T[:3, :3].copy()
    print("[ur10e_frax_demo] WARNING: phase1_results.npz not found — using identity base.")
    return np.zeros(3), np.eye(3)

# ---------------------------------------------------------------------------
# Frame note
# ---------------------------------------------------------------------------
# frax kinematics use the URDF frame: robot base at world origin, no rotation.
# The menagerie MuJoCo MJCF places the base body with quat="0 0 0 -1"
# (180° around Z), so MuJoCo world X/Y are negated relative to the frax frame.
#
#   frax → MuJoCo:  (x, y, z) → (-x, -y, z)
#   MuJoCo → frax:  (x, y, z) → (-x, -y, z)
#
# Obstacle positions are defined in the frax frame (CBF) and converted to
# MuJoCo world frame for visualization.

def _frax_to_mujoco(pos: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=float).copy()
    p[..., :2] *= -1
    return p


# ---------------------------------------------------------------------------
# Build frax collision data from oscbf ur_collision_model
# ---------------------------------------------------------------------------
def _make_frax_collision_data() -> dict:
    """Convert oscbf ur_collision_model to the dict format frax expects."""
    sys.path.insert(0, str(_ROOT / "main_logic"))
    from oscbf.core.ur_collision_model import (
        positions_list, radii_list,
        pairs_sc,
        base_position, base_radius, base_sc_idxs,
    )

    return {
        # Per-joint link spheres (oscbf link_i → frax joint i-1)
        "positions": positions_list,
        "radii": radii_list,
        # Fixed-base sphere (approximates the robot column)
        "root_positions": (base_position,),
        "root_radii": (base_radius,),
        # Root self-collision: each body sphere vs. the single base sphere (idx 0)
        "root_sc_pairs": tuple((0, body_idx) for body_idx in base_sc_idxs),
        "root_sc_tols": tuple(0.0 for _ in base_sc_idxs),
        # Body self-collision pairs (global flat sphere indices)
        "body_sc_pairs": pairs_sc,
        "body_sc_tols": tuple(0.0 for _ in pairs_sc),
    }


# ---------------------------------------------------------------------------
# Obstacles — defined in frax / URDF base frame
# ---------------------------------------------------------------------------
# 2 spheres + 2 yaw-rotated boxes. Box size is half-extents (MuJoCo convention).
OBSTACLES = {
    "sphere1": dict(pos=np.array([ 0.50,  0.30, 0.55]), radius=0.12,  shape="sphere"),
    "sphere2": dict(pos=np.array([ 0.20, -0.55, 0.75]), radius=0.10,  shape="sphere"),
    "box1":    dict(pos=np.array([-0.15,  0.60, 0.45]), size=np.array([0.10, 0.06, 0.10]),
                    yaw_deg= 30.0, shape="box"),
    "box2":    dict(pos=np.array([ 0.65, -0.15, 0.30]), size=np.array([0.12, 0.06, 0.08]),
                    yaw_deg=-45.0, shape="box"),
}

_SPHERE_OBS = {k: v for k, v in OBSTACLES.items() if v["shape"] == "sphere"}
_BOX_OBS    = {k: v for k, v in OBSTACLES.items() if v["shape"] == "box"}

OBS_SPHERE_POSITIONS = np.array([v["pos"]    for v in _SPHERE_OBS.values()])  # (Ns, 3)
OBS_SPHERE_RADII     = np.array([v["radius"] for v in _SPHERE_OBS.values()])  # (Ns,)

OBS_BOX_CENTERS = np.array([v["pos"]  for v in _BOX_OBS.values()])  # (Nb, 3)
OBS_BOX_HALVES  = np.array([v["size"] for v in _BOX_OBS.values()])  # (Nb, 3) half-extents

# R_box_world.T  →  rotates world-frame displacement into box local frame
_OBS_R_BW = np.array([
    ScipyR.from_euler("z", v["yaw_deg"], degrees=True).as_matrix().T
    for v in _BOX_OBS.values()
])  # (Nb, 3, 3)


# ---------------------------------------------------------------------------
# CBF config — joint limits + link-sphere vs. obstacle collision avoidance
# ---------------------------------------------------------------------------
@jax.tree_util.register_static
class UR10eOSCBFConfig(OSCBFVelocityConfig):
    def __init__(
        self,
        robot,
        z_floor: float,
        joint_pos_min,
        joint_pos_max,
        sphere_positions: tuple,  # tuple of (x,y,z) in world frame
        sphere_radii: tuple,
        box_centers: tuple,       # tuple of (x,y,z) in world frame
        box_halves: tuple,        # tuple of (hx,hy,hz) half-extents
        box_R_bw: tuple,          # tuple of 9-float row-major R_box_world.T matrices
        base_pos: tuple,          # (3,) robot base position in world frame
        base_R_flat: tuple,       # (9,) row-major rotation: world_R_base
    ):
        self.z_floor = float(z_floor)
        self.joint_pos_min = tuple(float(v) for v in joint_pos_min)
        self.joint_pos_max = tuple(float(v) for v in joint_pos_max)
        self.sphere_positions = tuple(tuple(float(v) for v in p) for p in sphere_positions)
        self.sphere_radii     = tuple(float(r) for r in sphere_radii)
        self.box_centers  = tuple(tuple(float(v) for v in c) for c in box_centers)
        self.box_halves   = tuple(tuple(float(v) for v in h) for h in box_halves)
        self.box_R_bw = tuple(
            tuple(float(v) for v in R.ravel()) for R in box_R_bw
        )
        self.base_pos    = tuple(float(v) for v in base_pos)
        self.base_R_flat = tuple(float(v) for v in base_R_flat)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]

        h_joint_low  = q - jnp.asarray(self.joint_pos_min)
        h_joint_high = jnp.asarray(self.joint_pos_max) - q

        # frax FK is in robot base frame → convert to world frame
        robot_pos_b, robot_rad = self.robot.link_collision_data(q)  # (N,3), (N,)
        base_R  = jnp.asarray(self.base_R_flat).reshape(3, 3)
        base_p  = jnp.asarray(self.base_pos)
        robot_pos = robot_pos_b @ base_R.T + base_p               # (N,3) world frame

        h_parts = [h_joint_low, h_joint_high]

        # Sphere obstacles
        if self.sphere_positions:
            sph_pos = jnp.asarray(self.sphere_positions)
            sph_rad = jnp.asarray(self.sphere_radii)
            deltas    = robot_pos[:, None, :] - sph_pos[None, :, :]
            radii_sum = robot_rad[:, None] + sph_rad[None, :]
            h_parts.append(jnp.linalg.norm(deltas, axis=-1).ravel() - radii_sum.ravel())

        # Box obstacles — yaw-rotated box SDF (ported from linux_controller2.py)
        if self.box_centers:
            obs_c = jnp.asarray(self.box_centers)
            obs_h = jnp.asarray(self.box_halves)
            R_bw  = jnp.asarray(self.box_R_bw).reshape(-1, 3, 3)
            d_world = robot_pos[:, None, :] - obs_c[None, :, :]
            d_box   = jnp.einsum("oij,soj->soi", R_bw, d_world)
            inside  = jnp.abs(d_box) - obs_h[None, :, :]
            sdf     = jnp.linalg.norm(jnp.maximum(inside, 0.0), axis=2)
            h_parts.append((sdf - robot_rad[:, None]).ravel())

        # Floor avoidance (world-frame Z)
        h_parts.append(robot_pos[:, 2] - robot_rad - self.z_floor)

        return jnp.concatenate(h_parts)

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# ---------------------------------------------------------------------------
# MuJoCo scene builder
# ---------------------------------------------------------------------------
def _add_gizmo(parent, prefix: str, ox: float, oy: float, oz: float,
               length: float = 0.12, radius: float = 0.005):
    """Append XYZ axis cylinders + white origin dot to an XML element."""
    import xml.etree.ElementTree as ET
    half = length / 2
    for axis, rgba, dx, dy, dz, ex, ey in [
        ("x", "1 0.15 0.15 1", half, 0,    0,    0,          np.pi/2),
        ("y", "0.15 1 0.15 1", 0,    half, 0,   -np.pi/2,    0),
        ("z", "0.2 0.5 1.0 1", 0,    0,    half,  0,          0),
    ]:
        g = ET.SubElement(parent, "geom")
        g.set("name", f"{prefix}_{axis}")
        g.set("type", "cylinder"); g.set("size", f"{radius} {half}")
        g.set("pos", f"{ox+dx} {oy+dy} {oz+dz}")
        if abs(ex) > 1e-6 or abs(ey) > 1e-6:
            g.set("euler", f"{ex:.6f} {ey:.6f} 0")
        g.set("rgba", rgba); g.set("contype", "0"); g.set("conaffinity", "0")
    dot = ET.SubElement(parent, "geom")
    dot.set("name", f"{prefix}_dot"); dot.set("type", "sphere"); dot.set("size", "0.010")
    dot.set("pos", f"{ox} {oy} {oz}"); dot.set("rgba", "1 1 1 1")
    dot.set("contype", "0"); dot.set("conaffinity", "0")


def _build_mujoco_model(
    urdf_path: str,
    target_pos: np.ndarray,
    base_pos_mj: np.ndarray,
    base_quat_mj: np.ndarray,   # MuJoCo (w,x,y,z) — full quat for the base body
) -> mujoco.MjModel:
    """Build MuJoCo model: OBJ meshes + floor + lights + target + obstacles + gizmos."""
    import xml.etree.ElementTree as ET

    urdf_dir = Path(urdf_path).parent
    mjcf_path = urdf_dir / "ur10e_with_obj.xml"

    if mjcf_path.exists():
        mjcf_xml = mjcf_path.read_text()
        mjcf_xml = mjcf_xml.replace('meshdir="assets"',
                                     f'meshdir="{urdf_dir / "assets"}"')
    else:
        spec = mujoco.MjSpec.from_file(urdf_path)
        mjcf_xml = spec.to_xml()
        mjcf_xml = mjcf_xml.replace('file="meshes/', f'file="{urdf_dir}/meshes/')

    root = ET.fromstring(mjcf_xml)

    # ---- Visual / lighting ------------------------------------------------
    visual = ET.SubElement(root, "visual")
    hl = ET.SubElement(visual, "headlight")
    hl.set("diffuse", "0.6 0.6 0.6"); hl.set("ambient", "0.3 0.3 0.3"); hl.set("specular", "0 0 0")
    ET.SubElement(visual, "rgba").set("haze", "0.15 0.25 0.35 1")

    # ---- Assets -----------------------------------------------------------
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    sky = ET.SubElement(assets, "texture")
    sky.set("type", "skybox"); sky.set("builtin", "gradient")
    sky.set("rgb1", "0.3 0.5 0.7"); sky.set("rgb2", "0 0 0")
    sky.set("width", "512"); sky.set("height", "3072")
    gt = ET.SubElement(assets, "texture")
    gt.set("type", "2d"); gt.set("name", "groundplane"); gt.set("builtin", "checker")
    gt.set("mark", "edge"); gt.set("rgb1", "0.2 0.3 0.4"); gt.set("rgb2", "0.1 0.2 0.3")
    gt.set("markrgb", "0.8 0.8 0.8"); gt.set("width", "300"); gt.set("height", "300")
    gm = ET.SubElement(assets, "material")
    gm.set("name", "groundplane"); gm.set("texture", "groundplane")
    gm.set("texuniform", "true"); gm.set("texrepeat", "5 5"); gm.set("reflectance", "0.2")

    # ---- Reposition robot base body ---------------------------------------
    wb = root.find("worldbody")
    base_body = wb.find("body[@name='base']")
    if base_body is not None:
        w, x, y, z = base_quat_mj
        base_body.set("pos", f"{base_pos_mj[0]:.6f} {base_pos_mj[1]:.6f} {base_pos_mj[2]:.6f}")
        base_body.set("quat", f"{w:.8f} {x:.8f} {y:.8f} {z:.8f}")

    # ---- Worldbody additions ----------------------------------------------
    light = ET.SubElement(wb, "light")
    light.set("pos", "0 0 3"); light.set("dir", "0 0 -1"); light.set("directional", "true")

    floor = ET.SubElement(wb, "geom")
    floor.set("name", "floor"); floor.set("size", "0 0 0.05")
    floor.set("type", "plane"); floor.set("material", "groundplane")

    # ---- Gizmo: world origin (0,0,0) in MuJoCo world frame ---------------
    _add_gizmo(wb, "gizmo_world", 0, 0, 0, length=0.15)

    # ---- Gizmo: robot base (in MuJoCo world frame) ------------------------
    _add_gizmo(wb, "gizmo_base",
               float(base_pos_mj[0]), float(base_pos_mj[1]), float(base_pos_mj[2]),
               length=0.15)

    # ---- Target mocap sphere (red) ----------------------------------------
    tp = _frax_to_mujoco(target_pos)
    tb = ET.SubElement(wb, "body")
    tb.set("name", "target"); tb.set("mocap", "true")
    tb.set("pos", f"{tp[0]} {tp[1]} {tp[2]}")
    tg = ET.SubElement(tb, "geom")
    tg.set("type", "sphere"); tg.set("size", "0.05")
    tg.set("rgba", "1 0.2 0.2 0.8"); tg.set("contype", "0"); tg.set("conaffinity", "0")

    # ---- Obstacles — convert frax frame → MuJoCo world frame --------------
    for name, obs in OBSTACLES.items():
        mj_pos = _frax_to_mujoco(obs["pos"])
        g = ET.SubElement(wb, "geom")
        g.set("name", name)
        g.set("pos", f"{mj_pos[0]} {mj_pos[1]} {mj_pos[2]}")
        g.set("contype", "0"); g.set("conaffinity", "0")
        if obs["shape"] == "sphere":
            g.set("type", "sphere")
            g.set("size", str(obs["radius"]))
            g.set("rgba", "0.2 0.5 0.9 0.7")
        else:  # box
            s = obs["size"]
            g.set("type", "box")
            g.set("size", f"{s[0]} {s[1]} {s[2]}")
            g.set("rgba", "0.9 0.5 0.1 0.8")
            # Yaw in frax frame → MuJoCo frame: negate X,Y is R_z(180°), so yaw_mj = yaw_frax + π
            yaw_mj_rad = np.deg2rad(obs.get("yaw_deg", 0.0)) + np.pi
            g.set("euler", f"0 0 {yaw_mj_rad:.6f}")

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


# ---------------------------------------------------------------------------
# MuJoCo simulation environment
# ---------------------------------------------------------------------------
class UR10eEnv:
    """Minimal MuJoCo environment for the UR10e, built from the URDF."""

    def __init__(
        self,
        urdf_path: str,
        base_pos_mj: np.ndarray,      # MuJoCo world frame
        base_quat_mj: np.ndarray,     # MuJoCo (w,x,y,z)
        base_pos: np.ndarray,         # frax world frame — for mocap→base conversion
        base_R: np.ndarray,           # frax world frame rotation matrix
        q_init: np.ndarray | None = None,
        target_pos_frax: np.ndarray | None = None,
        real_time: bool = True,
    ):
        if q_init is None:
            q_init = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
        if target_pos_frax is None:
            target_pos_frax = base_pos + np.array([0.5, 0.0, 0.5])

        self._base_pos = base_pos.copy()
        self._base_R   = base_R.copy()
        self.model = _build_mujoco_model(urdf_path, target_pos_frax, base_pos_mj, base_quat_mj)
        self.data  = mujoco.MjData(self.model)

        self._num_joints = 6
        self._mocap_id   = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        ]

        self.data.qpos[: self._num_joints] = q_init
        mujoco.mj_forward(self.model, self.data)

        self.dt   = self.model.opt.timestep
        self.t    = 0.0
        self.real_time  = real_time
        self._last_wall = time.time()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def get_joint_state(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[: self._num_joints],
            self.data.qvel[: self._num_joints],
        ])

    def get_desired_ee_state(self) -> np.ndarray:
        """Mocap pos (MuJoCo world) → frax world → robot base frame for OSC."""
        mj_pos      = self.data.mocap_pos[self._mocap_id].copy()
        world_frax  = _frax_to_mujoco(mj_pos)                       # frax world frame
        des_pos     = self._base_R.T @ (world_frax - self._base_pos) # robot base frame
        # Desired EE orientation in base frame: pointing down along base -Z
        des_rot = np.array([[1.0, 0.0,  0.0],
                            [0.0, -1.0, 0.0],
                            [0.0,  0.0, -1.0]])
        return np.array([*des_pos, *des_rot.ravel(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def apply_velocity_control(self, qdot: np.ndarray) -> None:
        self.data.qvel[: self._num_joints] = qdot

    def step(self) -> None:
        self.data.qpos[: self._num_joints] += (
            self.data.qvel[: self._num_joints] * self.dt
        )
        mujoco.mj_forward(self.model, self.data)
        self.t += self.dt
        self.viewer.sync()
        if self.real_time:
            elapsed = time.time() - self._last_wall
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
            self._last_wall = time.time()

    def close(self) -> None:
        self.viewer.close()


# ---------------------------------------------------------------------------
# OSC velocity controller
# ---------------------------------------------------------------------------
def make_osc_velocity_step(robot, cbf, kp_pos=50.0, kp_rot=20.0, kp_joint=10.0):
    kp_task = jnp.array([kp_pos] * 3 + [kp_rot] * 3)
    kp_jnt = kp_joint

    @jax.jit
    def step(q, z_ee_des):
        des_pos = z_ee_des[:3]
        des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
        des_vel = z_ee_des[12:15]
        des_omega = z_ee_des[15:18]

        M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
        pos = ee_tmat[:3, 3]
        rot = ee_tmat[:3, :3]

        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia
        N = jnp.eye(robot.num_joints) - J_bar @ J

        pos_err = pos - des_pos
        rot_err = orientation_error_3D(rot, des_rot)
        task_err = jnp.concatenate([pos_err, rot_err])

        twist_des = jnp.concatenate([des_vel, des_omega])
        u_nom = J_bar @ (twist_des - kp_task * task_err) - N @ (kp_jnt * q)

        return cbf.safety_filter(q, u_nom)

    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Load calibrated base pose ----------------------------------------
    base_pos, base_R = _load_base_transform()
    print(f"[ur10e_frax_demo] Robot base (world frame): pos={np.round(base_pos, 3)}")

    # Compute MuJoCo-frame equivalents
    # base body quat = quat(R_z(180°) @ base_R)   (geometric alignment + world rotation)
    R_z180 = ScipyR.from_euler("z", np.pi).as_matrix()
    base_R_mj_total = R_z180 @ base_R              # total rotation for the base body
    q_xyzw = ScipyR.from_matrix(base_R_mj_total).as_quat()   # scipy: (x,y,z,w)
    base_quat_mj = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # MuJoCo: (w,x,y,z)
    base_pos_mj  = _frax_to_mujoco(base_pos)

    # ---- Obstacle positions are in world frame (frax convention) ----------
    # (No change needed to OBSTACLES; they're already world-frame constants.)

    # ---- Build robot model ------------------------------------------------
    print("[ur10e_frax_demo] Building collision model from oscbf...")
    frax_collision_data = _make_frax_collision_data()

    print("[ur10e_frax_demo] Loading robot model...")
    robot = load_ur10e(_URDF, collision_data=frax_collision_data)
    print(f"[ur10e_frax_demo] {robot.num_joints} joints, "
          f"has_collision_data={robot.has_collision_data}")

    # Joint limits from URDF (clamp to sensible range)
    q_min = np.clip(np.array(robot.joint_lower_limits, dtype=float), -2 * np.pi, 0)
    q_max = np.clip(np.array(robot.joint_upper_limits, dtype=float),  0, 2 * np.pi)

    # CBF
    cbf_config = UR10eOSCBFConfig(
        robot=robot,
        z_floor=0.02,
        joint_pos_min=q_min,
        joint_pos_max=q_max,
        sphere_positions=tuple(map(tuple, OBS_SPHERE_POSITIONS)),
        sphere_radii=tuple(float(r) for r in OBS_SPHERE_RADII),
        box_centers=tuple(map(tuple, OBS_BOX_CENTERS)),
        box_halves=tuple(map(tuple, OBS_BOX_HALVES)),
        box_R_bw=_OBS_R_BW,
        base_pos=tuple(base_pos),
        base_R_flat=tuple(base_R.ravel()),
    )
    cbf = CBF.from_config(cbf_config)

    # JIT warm-up
    step = make_osc_velocity_step(robot, cbf)
    q_init = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])
    _ = step(q_init, np.zeros(18))
    print("[ur10e_frax_demo] JIT compiled. Starting simulation.")
    print("  Obstacles (world / frax frame):")
    for name, obs in OBSTACLES.items():
        extra = f" yaw={obs['yaw_deg']}°" if "yaw_deg" in obs else f" r={obs.get('radius')}"
        print(f"    {name}: pos={obs['pos']}{extra}")
    print("  Gizmo: white/small = world origin | larger = robot base")
    print("  Double-click target sphere → Ctrl+right-drag to move it.")

    # MuJoCo environment  (target starts 0.5 m in front of robot, 0.5 m up)
    target_pos_frax = base_pos + np.array([0.50, 0.0, 0.50])
    env = UR10eEnv(
        urdf_path=_URDF,
        base_pos_mj=base_pos_mj,
        base_quat_mj=base_quat_mj,
        base_pos=base_pos,
        base_R=base_R,
        q_init=q_init,
        target_pos_frax=target_pos_frax,
        real_time=True,
    )

    qdot_limits = np.deg2rad([120, 120, 180, 180, 180, 180])

    try:
        while env.viewer.is_running():
            q = env.get_joint_state()[:6]
            z_ee_des = env.get_desired_ee_state()

            qdot_cmd = np.array(step(q, z_ee_des))
            qdot_cmd = np.clip(qdot_cmd, -qdot_limits, qdot_limits)

            # Deadband: freeze if EE is within 1 cm of target (both in base frame)
            _, _, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
            if np.linalg.norm(z_ee_des[:3] - np.array(ee_tmat[:3, 3])) < 0.01:
                qdot_cmd[:] = 0.0

            env.apply_velocity_control(qdot_cmd)
            env.step()
    finally:
        env.close()


if __name__ == "__main__":
    main()
