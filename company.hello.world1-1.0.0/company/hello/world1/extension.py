"""
Spray painting extension for Isaac Sim 5.0  — consolidated clean version
=========================================================================

KEY FIXES vs the document shared by the user
---------------------------------------------
1.  FrozenParticles COLOR visible in RTX viewport
    -----------------------------------------------
    Isaac Sim 5.0 RTX ignores displayColor/displayOpacity primvars on
    UsdGeom.Points entirely.  The ONLY way to show a color is to bind a real
    UsdPreviewSurface material.  _ensure_points_material() creates
      /World/Looks/SprayParticleMaterial   for in-flight particles
      /World/Looks/FrozenParticleMaterial  for frozen dots
    and updates their diffuseColor every frame.  Called unconditionally on
    every stage-prim update so color always matches the picker.

2.  FrozenParticles SIZE matches SprayParticles
    ----------------------------------------------
    Both prims use the same UI slider value (self._p_particle_size).
    Points and widths are written in the SAME Sdf.ChangeBlock so Hydra
    always sees them with identical counts — eliminates the RTX error
    "Unexpected widthsCount of N for M points".

3.  widths / points count mismatch eliminated
    --------------------------------------------
    Root cause: widths were written in a separate USD operation after points.
    Hydra can read between the two writes → mismatch.
    Fix: wrap BOTH writes in  with Sdf.ChangeBlock():  so they are one
    atomic notification to Hydra.  Empty arrays are also written atomically
    when n == 0 so no stale widths linger from a previous frame.

4.  Canvas width / height configurable from UI
    ----------------------------------------------
    Two FloatDrag sliders under 🖼️ Canvas Size control the UV-space extents
    passed to the Warp kernel.  Adjust to match the physical face size of
    your /World/CanvasPlane.
"""

import os
import asyncio
import math
import numpy as np
import omni.ext
import omni.ui as ui
import omni.usd
import omni.kit.app
import warp as wp
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt

from .wall_size import get_plane_size_from_matrix

wp.init()

# ---------------------------------------------------------------------------
# Prim paths
# ---------------------------------------------------------------------------
ACTIVE_PARTICLES_PATH = "/World/SprayParticles"
FROZEN_PARTICLES_PATH = "/World/FrozenParticles"
NOZZLE_PRIM_PATH      = "/World/SprayNozzle"
CANVAS_PRIM_PATH      = "/World/CanvasPlane"
WALL_MATERIAL_PATH    = "/World/Looks/WallPaintMaterial"
SPRAY_MTL_PATH        = "/World/Looks/SprayParticleMaterial"
FROZEN_MTL_PATH       = "/World/Looks/FrozenParticleMaterial"

TEX_WIDTH  = 512
TEX_HEIGHT = 512
BLANK_GRAY = 220

# ---------------------------------------------------------------------------
# UI defaults
# ---------------------------------------------------------------------------
DEFAULT_NOZZLE_RADIUS   = 0.10
DEFAULT_NOZZLE_HEIGHT   = 0.50
DEFAULT_CONE_SPREAD_DEG = 5.0

DEFAULT_MAX_PARTICLES   = 1000
DEFAULT_EMIT_PER_TICK   = 20
DEFAULT_SPEED           = 5.0
DEFAULT_IMPACT_RADIUS   = 1
DEFAULT_PARTICLE_SIZE   = 0.02   # diameter in metres — shared by BOTH prims

DEFAULT_CANVAS_WIDTH    = 2.0    # world-space UV width  (set to match your plane)
DEFAULT_CANVAS_HEIGHT   = 2.0    # world-space UV height (set to match your plane)

DEFAULT_PUSH_INTERVAL   = 4

DEFAULT_R, DEFAULT_G, DEFAULT_B, DEFAULT_A = 1.0, 0.0, 0.0, 1.0


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------
@wp.kernel
def spray_paint_kernel(
    positions:    wp.array(dtype=wp.vec3),
    directions:   wp.array(dtype=wp.vec3),
    hit_flags:    wp.array(dtype=wp.uint8),
    frozen_pos:   wp.array(dtype=wp.vec3),
    wall_pos:     wp.vec3,
    wall_normal:  wp.vec3,
    wall_right:   wp.vec3,
    wall_up:      wp.vec3,
    wall_w:       float,
    wall_h:       float,
    tex_buffer:   wp.array(dtype=wp.uint8),
    tex_width:    int,
    tex_height:   int,
    dt:           float,
    speed:        float,
    splat_radius: int,
    color_r:      float,
    color_g:      float,
    color_b:      float,
    color_a:      float,
):
    tid = wp.tid()
    v   = directions[tid]
    if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
        return

    p      = positions[tid]
    next_p = p + v * speed * dt
    d1 = wp.dot(p      - wall_pos, wall_normal)
    d2 = wp.dot(next_p - wall_pos, wall_normal)

    if d1 * d2 <= 0.0 and d1 >= 0.0:
        denom  = d1 - d2
        t_frac = float(0.0)
        if denom != float(0.0):
            t_frac = d1 / denom
        hit_pos = p + v * speed * dt * t_frac
        frozen_pos[tid] = hit_pos

        delta   = hit_pos - wall_pos
        local_r = wp.dot(delta, wall_right)
        local_u = wp.dot(delta, wall_up)
        u_coord = (local_r / wall_w) + 0.5
        v_coord = (local_u / wall_h) + 0.5
        px = wp.int32(u_coord * float(tex_width))
        py = wp.int32((1.0 - v_coord) * float(tex_height))

        dy = -splat_radius
        while dy <= splat_radius:
            dx = -splat_radius
            while dx <= splat_radius:
                cx = px + dx
                cy = py + dy
                if cx >= 0 and cx < tex_width and cy >= 0 and cy < tex_height:
                    base = (cy * tex_width + cx) * 4
                    tex_buffer[base + 0] = wp.uint8(wp.int32(color_r * 255.0))
                    tex_buffer[base + 1] = wp.uint8(wp.int32(color_g * 255.0))
                    tex_buffer[base + 2] = wp.uint8(wp.int32(color_b * 255.0))
                    tex_buffer[base + 3] = wp.uint8(wp.int32(color_a * 255.0))
                dx = dx + 1
            dy = dy + 1

        hit_flags[tid]  = wp.uint8(1)
        directions[tid] = wp.vec3(0.0, 0.0, 0.0)
    else:
        positions[tid] = next_p


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------
class CompanyHelloWorld1Extension(omni.ext.IExt):

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def on_startup(self, ext_id):
        print("[Spray] Starting up …")
        self._running          = False
        self._frame            = 0
        self._ext_dir          = os.path.dirname(os.path.abspath(__file__))
        self._texture_provider = None

        # param cache — overwritten by _read_params() every tick
        self._p_nozzle_radius   = DEFAULT_NOZZLE_RADIUS
        self._p_nozzle_height   = DEFAULT_NOZZLE_HEIGHT
        self._p_cone_spread_deg = DEFAULT_CONE_SPREAD_DEG
        self._p_max_particles   = DEFAULT_MAX_PARTICLES
        self._p_emit_per_tick   = DEFAULT_EMIT_PER_TICK
        self._p_speed           = DEFAULT_SPEED
        self._p_impact_radius   = DEFAULT_IMPACT_RADIUS
        self._p_particle_size   = DEFAULT_PARTICLE_SIZE
        self._p_canvas_width    = DEFAULT_CANVAS_WIDTH
        self._p_canvas_height   = DEFAULT_CANVAS_HEIGHT
        self._p_push_interval   = DEFAULT_PUSH_INTERVAL

        # texture buffer
        self._tex_size     = TEX_WIDTH * TEX_HEIGHT * 4
        self._tex_np       = np.full(self._tex_size, BLANK_GRAY, dtype=np.uint8)
        self._tex_np[3::4] = 255
        self._tex_wp       = wp.array(self._tex_np, dtype=wp.uint8, copy=True)

        # particle buffers
        self._MAX        = DEFAULT_MAX_PARTICLES
        self._positions  = np.zeros((self._MAX, 3), dtype=np.float32)
        self._directions = np.zeros((self._MAX, 3), dtype=np.float32)
        self._pos_wp     = wp.array(self._positions,  dtype=wp.vec3)
        self._dir_wp     = wp.array(self._directions, dtype=wp.vec3)
        self._hit_np     = np.zeros(self._MAX, dtype=np.uint8)
        self._hit_wp     = wp.array(self._hit_np, dtype=wp.uint8)
        self._frozen_wp  = wp.zeros(self._MAX, dtype=wp.vec3)
        self._frozen_np  = np.zeros((self._MAX, 3), dtype=np.float32)
        self._frozen_zero = np.zeros((self._MAX, 3), dtype=np.float32)
        self._frozen_positions = []   # Gf.Vec3f list — grows on hit
        self._frozen_colors    = []   # Gf.Vec3f list — RGB at hit time
        self._current_particle = 0

        self._ensure_nozzle()
        self._setup_canvas_texture()
        self._build_ui()
        self._loop_task = asyncio.ensure_future(self._main_loop())
        print("[Spray] Startup complete.")

    def on_shutdown(self):
        print("[Spray] Shutting down …")
        self._running = False
        if hasattr(self, "_loop_task") and self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
        if hasattr(self, "_window") and self._window:
            self._window.destroy()
            self._window = None
        self._texture_provider = None
        print("[Spray] Shutdown complete.")

    # ------------------------------------------------------------------ #
    # Param helpers                                                        #
    # ------------------------------------------------------------------ #
    def _read_params(self):
        def _f(m):
            try:    return float(m.as_float)
            except: return float(m.get_value_as_float())
        def _i(m):
            try:    return int(m.as_int)
            except: return int(m.get_value_as_int())

        self._p_nozzle_radius   = max(0.01,  _f(self._m_nozzle_radius))
        self._p_nozzle_height   = max(0.05,  _f(self._m_nozzle_height))
        self._p_cone_spread_deg = max(0.1,   _f(self._m_cone_spread))
        self._p_emit_per_tick   = max(1,     _i(self._m_emit_rate))
        self._p_speed           = max(0.1,   _f(self._m_speed))
        self._p_impact_radius   = max(0,     _i(self._m_impact_radius))
        self._p_particle_size   = max(0.001, _f(self._m_particle_size))
        self._p_canvas_width    = max(0.1,   _f(self._m_canvas_width))
        self._p_canvas_height   = max(0.1,   _f(self._m_canvas_height))
        self._p_push_interval   = max(1,     _i(self._m_push_interval))

        new_max = max(10, min(2000, _i(self._m_max_particles)))
        if new_max != self._MAX:
            self._reallocate_particle_buffers(new_max)

    def _reallocate_particle_buffers(self, new_max: int):
        self._MAX         = new_max
        self._positions   = np.zeros((new_max, 3), dtype=np.float32)
        self._directions  = np.zeros((new_max, 3), dtype=np.float32)
        self._pos_wp      = wp.array(self._positions,  dtype=wp.vec3)
        self._dir_wp      = wp.array(self._directions, dtype=wp.vec3)
        self._hit_np      = np.zeros(new_max, dtype=np.uint8)
        self._hit_wp      = wp.array(self._hit_np, dtype=wp.uint8)
        self._frozen_wp   = wp.zeros(new_max, dtype=wp.vec3)
        self._frozen_np   = np.zeros((new_max, 3), dtype=np.float32)
        self._frozen_zero = np.zeros((new_max, 3), dtype=np.float32)
        self._current_particle = 0
        print(f"[Spray] Pool resized to {new_max}")

    def _get_color(self):
        if self._color_model is not None:
            try:
                items = self._color_model.get_item_children(None)
                return tuple(
                    self._color_model.get_item_value_model(c, 0).as_float
                    for c in items[:4]
                )
            except Exception:
                pass
        return (DEFAULT_R, DEFAULT_G, DEFAULT_B, DEFAULT_A)

    # ------------------------------------------------------------------ #
    # Scene helpers                                                        #
    # ------------------------------------------------------------------ #
    def _ensure_nozzle(self):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(NOZZLE_PRIM_PATH)
        if not prim.IsValid():
            cyl = UsdGeom.Cylinder.Define(stage, NOZZLE_PRIM_PATH)
            cyl.GetAxisAttr().Set("Z")
            UsdGeom.XformCommonAPI(cyl).SetTranslate((0, 0, 3))
            print("[Spray] Nozzle created.")
        self._apply_nozzle_dims()

    def _apply_nozzle_dims(self):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(NOZZLE_PRIM_PATH)
        if not prim or not prim.IsValid():
            return
        cyl = UsdGeom.Cylinder(prim)
        if cyl:
            cyl.GetRadiusAttr().Set(float(self._p_nozzle_radius))
            cyl.GetHeightAttr().Set(float(self._p_nozzle_height))

    def _setup_canvas_texture(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        canvas_prim = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
        if not canvas_prim or not canvas_prim.IsValid():
            print(f"[Spray] {CANVAS_PRIM_PATH} not found.")
            return

        # clear displayColor so it doesn't override the texture in RTX
        gprim = UsdGeom.Gprim(canvas_prim)
        dc    = gprim.GetDisplayColorAttr()
        if dc:
            dc.Clear()

        mtl_path = Sdf.Path(WALL_MATERIAL_PATH)
        mtl_prim = stage.GetPrimAtPath(mtl_path)
        if not mtl_prim or not mtl_prim.IsValid():
            mtl = UsdShade.Material.Define(stage, mtl_path)
            shd = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
            shd.CreateIdAttr("UsdPreviewSurface")
            shd.CreateInput("roughness",           Sdf.ValueTypeNames.Float).Set(0.4)
            shd.CreateInput("metallic",            Sdf.ValueTypeNames.Float).Set(0.0)
            shd.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
            mtl.CreateSurfaceOutput().ConnectToSource(shd.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(canvas_prim).Bind(mtl)
            shader = shd
        else:
            sp     = stage.GetPrimAtPath(mtl_path.AppendPath("Shader"))
            shader = UsdShade.Shader(sp) if sp and sp.IsValid() else None

        tx_path = mtl_path.AppendPath("DiffuseColorTx")
        tx_prim = stage.GetPrimAtPath(tx_path)
        tx_shd  = (UsdShade.Shader.Define(stage, tx_path)
                   if not tx_prim or not tx_prim.IsValid()
                   else UsdShade.Shader(tx_prim))
        tx_shd.CreateIdAttr("UsdUVTexture")

        fi = tx_shd.GetInput("file") or \
             tx_shd.CreateInput("file", Sdf.ValueTypeNames.Asset)
        fi.Set(Sdf.AssetPath("dynamic://WallPaintTexture"))

        cs = tx_shd.GetInput("sourceColorSpace") or \
             tx_shd.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token)
        cs.Set("sRGB")

        if not tx_shd.GetOutput("rgb"):
            tx_shd.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        if shader is not None:
            di = shader.GetInput("diffuseColor") or \
                 shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            di.ConnectToSource(tx_shd.ConnectableAPI(), "rgb")

        if self._texture_provider is None:
            self._texture_provider = ui.DynamicTextureProvider("WallPaintTexture")
        self._push_texture()
        print("[Spray] Canvas texture → dynamic://WallPaintTexture")

    # ------------------------------------------------------------------ #
    # Canvas / nozzle queries                                              #
    # ------------------------------------------------------------------ #
    def _get_canvas_vectors(self):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(CANVAS_PRIM_PATH)
        if not prim or not prim.IsValid():
            return None, None, None, None
        m   = omni.usd.get_world_transform_matrix(prim)
        pos = m.ExtractTranslation()
        rot = m.ExtractRotationMatrix()
        def _v(r): return np.array([r[0], r[1], r[2]], dtype=np.float32)
        return (
            np.array([pos[0], pos[1], pos[2]], dtype=np.float32),
            _v(rot.GetRow(0)),   # right  = local +X
            _v(rot.GetRow(1)),   # up     = local +Y
            _v(rot.GetRow(2)),   # normal = local +Z  (face normal confirmed)
        )

    def _get_canvas_size(self):
        # Use UI slider values directly — stable and no USD attribute reading
        return self._p_canvas_width, self._p_canvas_height

    def _get_prim_transform(self, prim_path):
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None, None
        m   = omni.usd.get_world_transform_matrix(prim)
        pos = m.ExtractTranslation()
        fwd = m.ExtractRotationMatrix().GetRow(2)
        return (
            np.array([pos[0], pos[1], pos[2]], dtype=np.float32),
            np.array([fwd[0], fwd[1], fwd[2]], dtype=np.float32),
        )

    # ------------------------------------------------------------------ #
    # Particle emit                                                        #
    # ------------------------------------------------------------------ #
    def _emit_particles(self, origin, base_direction):
        spread = math.tan(math.radians(self._p_cone_spread_deg))
        for _ in range(int(self._p_emit_per_tick)):
            i = self._current_particle % self._MAX
            self._positions[i]  = origin
            noise = (np.random.rand(3) - 0.5) * 2.0 * spread
            d     = base_direction + noise
            mag   = np.linalg.norm(d)
            if mag > 1e-6:
                d /= mag
            self._directions[i] = d.astype(np.float32)
            self._current_particle += 1
        self._pos_wp.assign(self._positions)
        self._dir_wp.assign(self._directions)

    # ------------------------------------------------------------------ #
    # Points prim writer                                                   #
    # The ONE function that writes to UsdGeom.Points.                     #
    # Points and widths are ALWAYS written together inside Sdf.ChangeBlock #
    # so Hydra receives a single atomic notification — counts never differ. #
    # ------------------------------------------------------------------ #
    def _write_points_prim(self, prim_path: str, pts: list, size: float):
        """
        Create-or-update a UsdGeom.Points prim at prim_path.

        pts  : list of Gf.Vec3f  (may be empty)
        size : sphere diameter in world units

        Writes points AND widths inside Sdf.ChangeBlock.
        Also ensures a material is bound so the prim shows a color in RTX.
        Does NOT set displayColor primvars — RTX ignores them for Points.
        """
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        n = len(pts)

        # Create prim on first call
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            prim = UsdGeom.Points.Define(stage, prim_path).GetPrim()

        geom   = UsdGeom.Points(prim)
        pts_vt = Vt.Vec3fArray(pts)
        wid_vt = Vt.FloatArray([size] * n)

        # Atomic write — Hydra sees one change notification, counts always match
        with Sdf.ChangeBlock():
            geom.GetPointsAttr().Set(pts_vt)
            w = geom.GetWidthsAttr()
            if not w or not w.IsValid():
                w = geom.CreateWidthsAttr()
            w.Set(wid_vt)
            geom.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

        print(f"[Spray:DBG] {prim_path}: pts={n} widths={n} size={size:.4f}m")

    # ------------------------------------------------------------------ #
    # Material helper — color for Points prims                            #
    # ------------------------------------------------------------------ #
    def _ensure_points_material(self, prim_path: str, mat_path: str,
                                 rgb: tuple):
        """
        Bind a UsdPreviewSurface material to prim_path and keep its
        diffuseColor in sync with rgb.  This is the only way to show
        color on UsdGeom.Points in Isaac Sim 5.0 RTX.
        """
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        r, g, b  = float(rgb[0]), float(rgb[1]), float(rgb[2])
        mtl_sdf  = Sdf.Path(mat_path)
        shd_path = mtl_sdf.AppendPath("Shader")
        mtl_prim = stage.GetPrimAtPath(mtl_sdf)

        if not mtl_prim or not mtl_prim.IsValid():
            # First time — create material and bind it
            mtl = UsdShade.Material.Define(stage, mtl_sdf)
            shd = UsdShade.Shader.Define(stage, shd_path)
            shd.CreateIdAttr("UsdPreviewSurface")
            shd.CreateInput("roughness",           Sdf.ValueTypeNames.Float).Set(1.0)
            shd.CreateInput("metallic",            Sdf.ValueTypeNames.Float).Set(0.0)
            shd.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
            shd.CreateInput("diffuseColor",
                            Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
            shd.CreateInput("emissiveColor",
                            Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
            mtl.CreateSurfaceOutput().ConnectToSource(
                shd.ConnectableAPI(), "surface")
            pts_prim = stage.GetPrimAtPath(prim_path)
            if pts_prim and pts_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(pts_prim).Bind(mtl)
                print(f"[Spray:DBG] material {mat_path} bound to {prim_path}")
        else:
            # Update color every frame
            shd_prim = stage.GetPrimAtPath(shd_path)
            if shd_prim and shd_prim.IsValid():
                shd = UsdShade.Shader(shd_prim)
                col = Gf.Vec3f(r, g, b)
                for inp_name in ("diffuseColor", "emissiveColor"):
                    inp = shd.GetInput(inp_name)
                    if not inp:
                        inp = shd.CreateInput(inp_name,
                                              Sdf.ValueTypeNames.Color3f)
                    inp.Set(col)

    # ------------------------------------------------------------------ #
    # Texture helpers                                                      #
    # ------------------------------------------------------------------ #
    def _push_texture(self):
        if self._texture_provider is None:
            self._texture_provider = ui.DynamicTextureProvider("WallPaintTexture")
        try:
            self._tex_np[:] = self._tex_wp.numpy()
            self._texture_provider.set_bytes_data(
                self._tex_np.tolist(), [TEX_WIDTH, TEX_HEIGHT])
        except Exception as e:
            print("[Spray] push_texture failed:", e)

    def _write_png(self, path):
        try:
            self._tex_np[:] = self._tex_wp.numpy()
            img = self._tex_np.reshape((TEX_HEIGHT, TEX_WIDTH, 4)).copy()
            try:
                import PIL.Image as PILImage
                PILImage.fromarray(img, "RGBA").save(path)
                return True
            except ImportError:
                pass
            try:
                import cv2
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
                return True
            except Exception:
                pass
            print("[Spray] Neither PIL nor cv2 available.")
        except Exception as e:
            print("[Spray] _write_png failed:", e)
        return False

    def _save_paint_image(self):
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(self._ext_dir, f"paint_saved_{stamp}.png")
        if self._write_png(path):
            print("[Spray] Saved:", path)
        return path

    # ------------------------------------------------------------------ #
    # Reset — UNCHANGED logic                                             #
    # ------------------------------------------------------------------ #
    def _reset(self):
        # 1 & 2 — blank texture on CPU, GPU, provider
        self._tex_np[:]    = BLANK_GRAY
        self._tex_np[3::4] = 255
        self._tex_wp.assign(self._tex_np)
        self._push_texture()

        # 3 — overwrite PNG on disk
        self._write_png(os.path.join(self._ext_dir, "paint_diffuse.png"))

        # 4 — zero active particles
        self._positions[:]  = 0.0
        self._directions[:] = 0.0
        self._pos_wp.assign(self._positions)
        self._dir_wp.assign(self._directions)

        # 5 — reset counter
        self._current_particle = 0

        # 6 — clear frozen dots
        self._frozen_positions = []
        self._frozen_colors    = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            fp = stage.GetPrimAtPath(FROZEN_PARTICLES_PATH)
            if fp and fp.IsValid():
                with Sdf.ChangeBlock():
                    g = UsdGeom.Points(fp)
                    g.GetPointsAttr().Set(Vt.Vec3fArray([]))
                    wa = g.GetWidthsAttr()
                    if wa and wa.IsValid():
                        wa.Set(Vt.FloatArray([]))

        print("[Spray] Reset complete.")

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    async def _main_loop(self):
        app       = omni.kit.app.get_app()
        prev_time = None
        while True:
            await app.next_update_async()
            if not self._running:
                prev_time = None
                continue
            curr      = app.get_time_since_start_s()
            dt        = (0.016 if prev_time is None
                         else max(0.001, min(curr - prev_time, 0.05)))
            prev_time = curr
            self._tick(dt)

    # ------------------------------------------------------------------ #
    # Per-frame tick                                                       #
    # ------------------------------------------------------------------ #
    def _tick(self, dt: float):
        # 0. Pull UI values into param cache
        self._read_params()
        self._apply_nozzle_dims()

        # 1. Nozzle
        nozzle_pos, nozzle_dir = self._get_prim_transform(NOZZLE_PRIM_PATH)
        if nozzle_pos is None:
            return

        # 2. Canvas
        c_pos, c_right, c_up, c_normal = self._get_canvas_vectors()
        self._emit_particles(nozzle_pos, nozzle_dir)
        if c_pos is None:
            return

        canvas_w, canvas_h = self._get_canvas_size()
        r, g, b, a         = self._get_color()

        # 3. Clear hit flags and frozen output buffer before kernel
        self._hit_np[:] = 0
        self._hit_wp.assign(self._hit_np)
        self._frozen_wp.assign(self._frozen_zero)

        # 4. Warp kernel — 1000 threads in parallel on GPU
        wp.launch(
            kernel=spray_paint_kernel,
            dim=self._MAX,
            inputs=[
                self._pos_wp, self._dir_wp, self._hit_wp, self._frozen_wp,
                wp.vec3(float(c_pos[0]),    float(c_pos[1]),    float(c_pos[2])),
                wp.vec3(float(c_normal[0]), float(c_normal[1]), float(c_normal[2])),
                wp.vec3(float(c_right[0]),  float(c_right[1]),  float(c_right[2])),
                wp.vec3(float(c_up[0]),     float(c_up[1]),     float(c_up[2])),
                float(canvas_w), float(canvas_h),
                self._tex_wp, TEX_WIDTH, TEX_HEIGHT,
                float(dt), float(self._p_speed), int(self._p_impact_radius),
                float(r), float(g), float(b), float(a),
            ],
        )
        wp.synchronize()

        # 5. Pull results back to CPU
        self._positions[:] = self._pos_wp.numpy()
        self._hit_np[:]    = self._hit_wp.numpy()
        self._frozen_np[:] = self._frozen_wp.numpy()

        # 6. Accumulate frozen dots; recycle hit slots
        hit_indices = np.where(self._hit_np == 1)[0]
        if len(hit_indices) > 0:
            print(f"[Spray:DBG] {len(hit_indices)} hits  total_frozen={len(self._frozen_positions)+len(hit_indices)}")
            for idx in hit_indices:
                hp = self._frozen_np[idx]
                if hp[0] == 0.0 and hp[1] == 0.0 and hp[2] == 0.0:
                    print(f"[Spray:DBG] SKIP zero hit slot {idx}")
                    continue
                self._frozen_positions.append(
                    Gf.Vec3f(float(hp[0]), float(hp[1]), float(hp[2])))
                self._frozen_colors.append(
                    Gf.Vec3f(float(r), float(g), float(b)))
            self._directions[hit_indices] = 0.0
            self._dir_wp.assign(self._directions)

        # 7. Update stage prims every other frame
        self._frame += 1
        if self._frame % 2 == 0:
            psize = float(self._p_particle_size)

            # --- Active in-flight particles ---
            # Filter on direction==0 only (position-zero guard removed —
            # nozzle may be near world origin so position==0 is valid)
            active_pts = [
                Gf.Vec3f(float(self._positions[i, 0]),
                         float(self._positions[i, 1]),
                         float(self._positions[i, 2]))
                for i in range(self._MAX)
                if not (self._directions[i, 0] == 0.0 and
                        self._directions[i, 1] == 0.0 and
                        self._directions[i, 2] == 0.0)
            ]
            # Write points+widths atomically
            self._write_points_prim(ACTIVE_PARTICLES_PATH, active_pts, psize)
            # Bind/update material so spray particles show paint color in RTX
            if active_pts:
                self._ensure_points_material(
                    ACTIVE_PARTICLES_PATH, SPRAY_MTL_PATH, (r, g, b))

            # --- Frozen dots — nudged 1 mm above canvas surface ---
            NUDGE = 0.001
            cn = c_normal
            nudged = [
                Gf.Vec3f(
                    p[0] + float(cn[0]) * NUDGE,
                    p[1] + float(cn[1]) * NUDGE,
                    p[2] + float(cn[2]) * NUDGE,
                )
                for p in self._frozen_positions
            ]
            # Write points+widths atomically — SAME psize as spray particles
            self._write_points_prim(FROZEN_PARTICLES_PATH, nudged, psize)
            # Bind/update material for frozen dots
            if nudged:
                last = (self._frozen_colors[-1]
                        if self._frozen_colors
                        else Gf.Vec3f(r, g, b))
                self._ensure_points_material(
                    FROZEN_PARTICLES_PATH, FROZEN_MTL_PATH,
                    (float(last[0]), float(last[1]), float(last[2])))

        # 8. Push texture to DynamicTextureProvider periodically
        if self._frame % max(1, int(self._p_push_interval)) == 0:
            self._push_texture()

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self._color_model     = None
        self._m_nozzle_radius = None
        self._m_nozzle_height = None
        self._m_cone_spread   = None
        self._m_max_particles = None
        self._m_emit_rate     = None
        self._m_speed         = None
        self._m_particle_size = None
        self._m_canvas_width  = None
        self._m_canvas_height = None
        self._m_impact_radius = None
        self._m_push_interval = None

        self._window = ui.Window("Dynamic Spray Painter", width=400, height=860)
        with self._window.frame:
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            ):
                with ui.VStack(spacing=6, style={"margin": 8}):

                    ui.Label(
                        "Canvas: /World/CanvasPlane   Nozzle: /World/SprayNozzle",
                        style={"color": 0xFFAAAAAA, "font_size": 12},
                    )

                    # Paint color
                    self._section("🎨  Paint Color")
                    with ui.HStack(height=28):
                        ui.Label("RGBA :", width=70)
                        self._color_model = ui.ColorWidget(
                            DEFAULT_R, DEFAULT_G, DEFAULT_B, DEFAULT_A,
                            width=ui.Fraction(1), height=28,
                        ).model

                    ui.Separator()

                    # Nozzle
                    self._section("🔧  Nozzle")
                    self._m_nozzle_radius = self._float_row(
                        "Radius (m)", DEFAULT_NOZZLE_RADIUS, 0.01, 1.0, 0.01,
                        "Physical radius of the SprayNozzle cylinder prim")
                    self._m_nozzle_height = self._float_row(
                        "Height (m)", DEFAULT_NOZZLE_HEIGHT, 0.05, 5.0, 0.05,
                        "Physical height of the SprayNozzle cylinder prim")
                    self._m_cone_spread = self._float_row(
                        "Cone spread (°)", DEFAULT_CONE_SPREAD_DEG, 0.1, 60.0, 0.5,
                        "Half-angle of spray cone. Larger = wider cloud")

                    ui.Separator()

                    # Canvas size
                    self._section("🖼️  Canvas Size (UV mapping)")
                    self._m_canvas_width = self._float_row(
                        "Canvas width (m)", DEFAULT_CANVAS_WIDTH, 0.1, 20.0, 0.05,
                        "World-space width of the canvas face.\n"
                        "Set to match the actual width of /World/CanvasPlane.")
                    self._m_canvas_height = self._float_row(
                        "Canvas height (m)", DEFAULT_CANVAS_HEIGHT, 0.1, 20.0, 0.05,
                        "World-space height of the canvas face.\n"
                        "Set to match the actual height of /World/CanvasPlane.")

                    ui.Separator()

                    # Particles
                    self._section("💨  Particles")
                    self._m_max_particles = self._int_row(
                        "Max particles", DEFAULT_MAX_PARTICLES, 10, 2000, 10,
                        "Round-robin pool size")
                    self._m_emit_rate = self._int_row(
                        "Emit per tick", DEFAULT_EMIT_PER_TICK, 1, 200, 1,
                        "Particles fired per frame while holding Spray")
                    self._m_speed = self._float_row(
                        "Speed multiplier", DEFAULT_SPEED, 0.1, 50.0, 0.5,
                        "Scales particle velocity toward the canvas")
                    self._m_particle_size = self._float_row(
                        "Particle display size (m)", DEFAULT_PARTICLE_SIZE,
                        0.001, 0.5, 0.001,
                        "Sphere diameter for BOTH spray particles AND frozen dots.\n"
                        "Single slider keeps both sizes identical.")

                    ui.Separator()

                    # Impact
                    self._section("💥  Impact")
                    self._m_impact_radius = self._int_row(
                        "Splat radius (px)", DEFAULT_IMPACT_RADIUS, 0, 20, 1,
                        "Paint splat half-size in texture pixels.\n"
                        "0=1×1  1=3×3  2=5×5  5=11×11")

                    ui.Separator()

                    # Performance
                    self._section("⚡  Performance")
                    self._m_push_interval = self._int_row(
                        "Texture push interval", DEFAULT_PUSH_INTERVAL, 1, 60, 1,
                        "Push texture to viewport every N frames")

                    ui.Separator()

                    # Hold-to-spray
                    spray_btn = ui.Button(
                        "🎨  Hold to Spray", height=52,
                        style={"font_size": 17,
                               "background_color": 0xFF1A5276,
                               "color": 0xFFFFFFFF},
                    )
                    spray_btn.set_mouse_pressed_fn(
                        lambda x, y, b, m: setattr(self, "_running", True))
                    spray_btn.set_mouse_released_fn(
                        lambda x, y, b, m: setattr(self, "_running", False))

                    with ui.HStack(spacing=6, height=36):
                        ui.Button("💾  Save image",
                                  clicked_fn=self._save_paint_image)
                        ui.Button("🔄  Reset canvas",
                                  clicked_fn=self._reset,
                                  style={"background_color": 0xFF5D3A1A,
                                         "color": 0xFFFFFFFF})

    def _section(self, label):
        ui.Label(label, style={"font_size": 14, "color": 0xFFDDDDDD})

    def _float_row(self, label, default, lo, hi, step,
                   tooltip="") -> ui.AbstractValueModel:
        with ui.HStack(height=24, tooltip=tooltip):
            ui.Label(label, width=ui.Percent(56),
                     style={"color": 0xFFCCCCCC})
            w = ui.FloatDrag(min=lo, max=hi, step=step,
                             width=ui.Fraction(1), height=22)
            w.model.set_value(default)
        return w.model

    def _int_row(self, label, default, lo, hi, step,
                 tooltip="") -> ui.AbstractValueModel:
        with ui.HStack(height=24, tooltip=tooltip):
            ui.Label(label, width=ui.Percent(56),
                     style={"color": 0xFFCCCCCC})
            w = ui.IntDrag(min=lo, max=hi, step=step,
                           width=ui.Fraction(1), height=22)
            w.model.set_value(default)
        return w.model
