"""
create_prim.py
--------------
Manages the CanvasPlane prim and its paint texture.

Key design decisions
--------------------
* Target prim  : /World/CanvasPlane  (Cube, normal = local +X)
* Material     : /World/Looks/WallPaintMaterial already exists and is already
                 bound to the canvas — we ONLY rewire its DiffuseColorTx node
                 to use the  dynamic://  URI so DynamicTextureProvider drives
                 the live paint updates.  We never rebind or recreate the
                 material from scratch.
* Wall axes    : normal  = local +X (row 0 of rotation matrix)
                 right   = local +Y (row 1)  — horizontal on the canvas face
                 up      = local +Z (row 2)  — vertical   on the canvas face
* Wall size    : scale_Y × cube_size  (width)
                 scale_Z × cube_size  (height)
"""

import os
import numpy as np
import omni.usd
import omni.ui as ui

from pxr import UsdGeom, UsdShade, Sdf, UsdPhysics
from .wall_size import get_wall_size_xy_from_matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANVAS_PRIM_PATH    = "/World/CanvasPlane"
WALL_MATERIAL_PATH  = "/World/Looks/WallPaintMaterial"

DYNAMIC_TEX_NAME    = "WallPaintTexture"
DYNAMIC_TEX_URI     = f"dynamic://{DYNAMIC_TEX_NAME}"

TEX_WIDTH           = 512
TEX_HEIGHT          = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_canvas_texture(texture_provider_ref: list) -> None:
    """
    Rewire the existing WallPaintMaterial's DiffuseColorTx node to use
    the dynamic:// URI and create the DynamicTextureProvider.

    Must be called once (e.g. at extension startup or when the user clicks
    'Setup Canvas').

    Parameters
    ----------
    texture_provider_ref : one-element list used as an output — receives the
                           DynamicTextureProvider instance.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[Canvas] No stage available.")
        return

    canvas_prim = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
    if not canvas_prim or not canvas_prim.IsValid():
        print(f"[Canvas] Prim {CANVAS_PRIM_PATH} not found in stage.")
        return

    _rewire_material_to_dynamic(stage, texture_provider_ref)

    # Push an initial light-gray canvas so the wall is not black on first render
    tex_size        = TEX_WIDTH * TEX_HEIGHT * 4
    initial_tex     = np.full(tex_size, 220, dtype=np.uint8)
    initial_tex[3::4] = 255
    push_texture(initial_tex, texture_provider_ref)
    print("[Canvas] Texture setup complete.")


def get_canvas_vectors():
    """
    Returns (pos, right, up, normal) as numpy float32 arrays.
    For a Cube with local +X as its face normal:
      normal = local +X = rotation row 0
      right  = local +Y = rotation row 1   (horizontal on face)
      up     = local +Z = rotation row 2   (vertical   on face)

    Returns (None, None, None, None) if the prim is missing.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None, None, None

    prim = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
    if not prim or not prim.IsValid():
        return None, None, None, None

    world_mat = omni.usd.get_world_transform_matrix(prim)
    pos = world_mat.ExtractTranslation()
    rot = world_mat.ExtractRotationMatrix()

    def _v(row):
        return np.array([row[0], row[1], row[2]], dtype=np.float32)

    return (
        np.array([pos[0], pos[1], pos[2]], dtype=np.float32),
        _v(rot.GetRow(1)),   # right  = local Y
        _v(rot.GetRow(2)),   # up     = local Z
        _v(rot.GetRow(0)),   # normal = local X  ← face normal
    )


def get_canvas_size_wh():
    """
    Returns (width, height) in world-space units for the canvas face.
    Width  corresponds to local Y extent.
    Height corresponds to local Z extent.
    Defaults to (5.0, 5.0) if the prim is unavailable.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 5.0, 5.0

    prim = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
    if not prim or not prim.IsValid():
        return 5.0, 5.0

    cube      = UsdGeom.Cube(prim)
    size_attr = cube.GetSizeAttr() if cube else None
    size      = size_attr.Get() if size_attr else None
    if size is None:
        size = 1.0

    world_mat              = omni.usd.get_world_transform_matrix(prim)
    _sx, scale_y, scale_z  = get_wall_size_xy_from_matrix(world_mat, size)
    return scale_y, scale_z   # width=Y, height=Z


def push_texture(tex_np: np.ndarray, provider_ref: list) -> None:
    """
    Push flat uint8 RGBA numpy array to the DynamicTextureProvider.

    Parameters
    ----------
    tex_np       : flat array, length TEX_WIDTH * TEX_HEIGHT * 4, dtype uint8
    provider_ref : one-element list holding the DynamicTextureProvider
    """
    provider = provider_ref[0]
    if provider is None:
        return
    try:
        provider.set_bytes_data(tex_np.tolist(), [TEX_WIDTH, TEX_HEIGHT])
    except Exception as e:
        print("[Canvas] Failed to push texture:", e)


def save_paint_image(tex_np: np.ndarray, save_dir: str):
    """Write the current paint buffer to a timestamped PNG file."""
    try:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(save_dir, f"paint_saved_{stamp}.png")
        img   = tex_np.reshape((TEX_HEIGHT, TEX_WIDTH, 4)).copy()

        try:
            import PIL.Image as PILImage
            PILImage.fromarray(img, "RGBA").save(path)
            print("[Canvas] Saved:", path)
            return path
        except ImportError:
            pass
        try:
            import cv2
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
            print("[Canvas] Saved:", path)
            return path
        except Exception:
            pass

        print("[Canvas] Neither PIL nor cv2 available — image not saved.")
    except Exception as e:
        print("[Canvas] Save failed:", e)
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rewire_material_to_dynamic(stage, texture_provider_ref: list) -> None:
    """
    Find the existing WallPaintMaterial, locate (or create) its
    DiffuseColorTx UsdUVTexture node, and point its 'file' input at
    the dynamic:// URI so DynamicTextureProvider drives live updates.
    """
    mtl_path   = Sdf.Path(WALL_MATERIAL_PATH)
    mtl_prim   = stage.GetPrimAtPath(mtl_path)

    if not mtl_prim or not mtl_prim.IsValid():
        print("[Canvas] WallPaintMaterial not found — creating it.")
        _create_material_from_scratch(stage, texture_provider_ref)
        return

    # Try to find an existing UsdUVTexture child named DiffuseColorTx
    tex_path  = mtl_path.AppendPath("DiffuseColorTx")
    tex_prim  = stage.GetPrimAtPath(tex_path)

    if not tex_prim or not tex_prim.IsValid():
        # Create the texture node under the existing material
        tex_shader = UsdShade.Shader.Define(stage, tex_path)
        tex_shader.CreateIdAttr("UsdUVTexture")
    else:
        tex_shader = UsdShade.Shader(tex_prim)

    # Point the file input at dynamic://
    file_input = tex_shader.GetInput("file")
    if not file_input:
        file_input = tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_input.Set(Sdf.AssetPath(DYNAMIC_TEX_URI))

    # Ensure rgb output exists and wire it to the surface shader diffuseColor
    if not tex_shader.GetOutput("rgb"):
        tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # Find the surface shader (conventionally named "Shader") and connect
    shader_path = mtl_path.AppendPath("Shader")
    shader_prim = stage.GetPrimAtPath(shader_path)
    if shader_prim and shader_prim.IsValid():
        surface_shader = UsdShade.Shader(shader_prim)
        diff_input     = surface_shader.GetInput("diffuseColor")
        if not diff_input:
            from pxr import Sdf as _Sdf
            diff_input = surface_shader.CreateInput(
                "diffuseColor", _Sdf.ValueTypeNames.Color3f
            )
        diff_input.ConnectToSource(tex_shader.ConnectableAPI(), "rgb")

    # Create the DynamicTextureProvider (idempotent — reuse if already set)
    if texture_provider_ref[0] is None:
        texture_provider_ref[0] = ui.DynamicTextureProvider(DYNAMIC_TEX_NAME)
        print(f"[Canvas] DynamicTextureProvider '{DYNAMIC_TEX_NAME}' created.")


def _create_material_from_scratch(stage, texture_provider_ref: list) -> None:
    """Fallback: build a full UsdPreviewSurface material and bind it."""
    from pxr import UsdShade, Sdf, Gf

    canvas_prim = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
    if not canvas_prim or not canvas_prim.IsValid():
        return

    mtl_path = Sdf.Path(WALL_MATERIAL_PATH)
    mtl      = UsdShade.Material.Define(stage, mtl_path)

    shader_path = mtl_path.AppendPath("Shader")
    shader      = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic",  Sdf.ValueTypeNames.Float).Set(0.0)
    mtl.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    tex_path   = mtl_path.AppendPath("DiffuseColorTx")
    diffuse_tx = UsdShade.Shader.Define(stage, tex_path)
    diffuse_tx.CreateIdAttr("UsdUVTexture")
    diffuse_tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(DYNAMIC_TEX_URI)
    )
    diffuse_tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse_tx.ConnectableAPI(), "rgb"
    )

    UsdShade.MaterialBindingAPI.Apply(canvas_prim).Bind(mtl)

    if texture_provider_ref[0] is None:
        texture_provider_ref[0] = ui.DynamicTextureProvider(DYNAMIC_TEX_NAME)
        print(f"[Canvas] DynamicTextureProvider '{DYNAMIC_TEX_NAME}' created.")
