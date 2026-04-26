"""
nozzle.py
---------
Helpers for the SprayNozzle prim.

Coordinate contract
-------------------
The nozzle cylinder has  axis="Z"  in USD, so its local +Z is the barrel
direction.  In your scene the nozzle is rotated so that local +Z maps to
world  −X  (i.e. it points at the CanvasPlane).  We read the world-space
forward vector directly from the transform matrix row 2, so no hard-coded
direction is needed here — whatever the nozzle is rotated to in the viewport
is automatically the spray direction.
"""

import numpy as np
import omni.usd
from pxr import UsdGeom

NOZZLE_PRIM_PATH = "/World/SprayNozzle"


def ensure_nozzle_exists(prim_path: str = NOZZLE_PRIM_PATH) -> None:
    """Create the nozzle cylinder prim if it does not already exist."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        return
    cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
    cylinder.GetRadiusAttr().Set(0.1)
    cylinder.GetHeightAttr().Set(0.5)
    cylinder.GetAxisAttr().Set("Z")
    UsdGeom.XformCommonAPI(cylinder).SetTranslate((0, 100, 0))
    print(f"[Nozzle] Created nozzle prim at {prim_path}")


def get_nozzle_transform(prim_path: str = NOZZLE_PRIM_PATH):
    """
    Returns (position, forward_direction) as numpy float32 arrays,
    or (None, None) if the prim is missing.

    `forward_direction` is the local +Z axis expressed in world space —
    this is the direction particles are emitted.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None, None

    world_mat = omni.usd.get_world_transform_matrix(prim)
    pos     = world_mat.ExtractTranslation()
    forward = world_mat.ExtractRotationMatrix().GetRow(2)   # local +Z in world

    return (
        np.array([pos[0],     pos[1],     pos[2]],     dtype=np.float32),
        np.array([forward[0], forward[1], forward[2]], dtype=np.float32),
    )
