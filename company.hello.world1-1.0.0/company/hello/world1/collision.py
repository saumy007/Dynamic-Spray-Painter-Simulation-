"""
collision.py
------------
Warp-accelerated particle simulation and wall-collision / paint logic.

Fixes vs previous version
--------------------------
1. Hit detection uses a dedicated  wp.array(dtype=wp.uint8)  hit-flag array
   instead of a float sentinel in directions[].x — eliminates the float
   equality fragility that caused paint to stop after the first press.

2. Slot recycling is correct:  after every kernel launch the CPU reads the
   hit-flag array, returns hit slots to _free_slots, and zeroes them on both
   CPU and GPU before the next launch.

3. Paint color is passed into the kernel as four separate float inputs so the
   UI color picker can change it live without recompiling the kernel.

4. The d1 >= 0.0 back-face guard prevents particles that start behind the
   canvas from generating spurious splats.

5. dt is clamped to [0.001 … 0.05] seconds — no wall overshooting.

Canvas axis contract (set by create_prim.py)
--------------------------------------------
  wall_normal = local +X of CanvasPlane   (face the nozzle points at)
  wall_right  = local +Y                  (horizontal direction on canvas)
  wall_up     = local +Z                  (vertical   direction on canvas)
  wall_w      = world Y extent            (width)
  wall_h      = world Z extent            (height)
"""

import numpy as np
import warp as wp
import omni.usd
from pxr import UsdGeom, Gf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARTICLES_PRIM_PATH = "/World/SprayParticles"
MAX_PARTICLES       = 1000
EMIT_PER_TICK       = 20

wp.init()


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------
@wp.kernel
def spray_paint_kernel(
    positions:   wp.array(dtype=wp.vec3),
    directions:  wp.array(dtype=wp.vec3),
    hit_flags:   wp.array(dtype=wp.uint8),   # out: 1 = hit this frame
    wall_pos:    wp.vec3,
    wall_normal: wp.vec3,
    wall_right:  wp.vec3,
    wall_up:     wp.vec3,
    wall_w:      float,
    wall_h:      float,
    tex_buffer:  wp.array(dtype=wp.uint8),
    tex_width:   int,
    tex_height:  int,
    dt:          float,
    color_r:     float,
    color_g:     float,
    color_b:     float,
    color_a:     float,
):
    tid = wp.tid()

    # Skip dead / uninitialized slots (direction == zero vector)
    v = directions[tid]
    if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
        return

    p      = positions[tid]
    next_p = p + v * dt

    # Signed distances to the canvas plane before and after the step
    d1 = wp.dot(p      - wall_pos, wall_normal)
    d2 = wp.dot(next_p - wall_pos, wall_normal)

    # --- Hit: particle crossed the plane AND was in front of it ---
    if d1 * d2 <= 0.0 and d1 >= 0.0:
        # Exact intersection point
        denom  = d1 - d2
        t_frac = float(0.0)
        if denom != float(0.0):
            t_frac = d1 / denom
        hit_pos = p + v * dt * t_frac

        # Project hit into canvas-local 2-D space (origin at canvas centre)
        delta   = hit_pos - wall_pos
        local_r = wp.dot(delta, wall_right)   # horizontal
        local_u = wp.dot(delta, wall_up)      # vertical

        # Map to UV [0, 1]
        u_coord = (local_r / wall_w) + 0.5
        v_coord = (local_u / wall_h) + 0.5

        # UV → pixel; flip V for image Y-down convention
        px = wp.int32(u_coord * float(tex_width))
        py = wp.int32((1.0 - v_coord) * float(tex_height))

        # 3×3 splat
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx = px + dx
                cy = py + dy
                if cx >= 0 and cx < tex_width and cy >= 0 and cy < tex_height:
                    idx = (cy * tex_width + cx) * 4
                    tex_buffer[idx + 0] = wp.uint8(wp.int32(color_r * 255.0))
                    tex_buffer[idx + 1] = wp.uint8(wp.int32(color_g * 255.0))
                    tex_buffer[idx + 2] = wp.uint8(wp.int32(color_b * 255.0))
                    tex_buffer[idx + 3] = wp.uint8(wp.int32(color_a * 255.0))

        # Flag this slot for recycling — CPU will zero it and return to pool
        hit_flags[tid] = wp.uint8(1)

    else:
        # Particle still in flight
        positions[tid] = next_p


# ---------------------------------------------------------------------------
# ParticleSystem
# ---------------------------------------------------------------------------
class ParticleSystem:
    """
    Full lifecycle management for spray particles.

    Parameters
    ----------
    tex_buffer_wp : wp.array(dtype=wp.uint8)
        Shared Warp RGBA texture buffer (flat, length W*H*4).
    tex_width / tex_height : int
        Texture dimensions.
    """

    def __init__(self, tex_buffer_wp: wp.array, tex_width: int, tex_height: int):
        self._tex_wp     = tex_buffer_wp
        self._tex_w      = tex_width
        self._tex_h      = tex_height

        # CPU-side particle buffers
        self._pos_np  = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)
        self._dir_np  = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)

        # GPU mirrors
        self._pos_wp  = wp.array(self._pos_np, dtype=wp.vec3)
        self._dir_wp  = wp.array(self._dir_np, dtype=wp.vec3)

        # Hit-flag array — kernel writes 1 to signal a hit; CPU reads and recycles
        self._hit_np  = np.zeros(MAX_PARTICLES, dtype=np.uint8)
        self._hit_wp  = wp.array(self._hit_np, dtype=wp.uint8)

        # Slot management
        self._free_slots   = list(range(MAX_PARTICLES))
        self._active_slots = set()

        self._stage_frame  = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def emit(self, origin: np.ndarray, base_direction: np.ndarray) -> None:
        """
        Fire up to EMIT_PER_TICK particles from `origin`.
        Called every frame while the spray button is held.
        """
        if not self._free_slots:
            return   # pool exhausted — silently skip, no crash

        new_slots = []
        for _ in range(EMIT_PER_TICK):
            if not self._free_slots:
                break
            slot = self._free_slots.pop()
            self._active_slots.add(slot)
            new_slots.append(slot)

            self._pos_np[slot] = origin
            noise = (np.random.rand(3) - 0.5) * 0.1
            d     = base_direction + noise
            mag   = np.linalg.norm(d)
            if mag > 1e-6:
                d /= mag
            self._dir_np[slot] = d.astype(np.float32)

        # Upload full arrays — Warp does not yet support partial slice uploads
        self._pos_wp.assign(self._pos_np)
        self._dir_wp.assign(self._dir_np)

    def tick(
        self,
        wall_pos:    np.ndarray,
        wall_right:  np.ndarray,
        wall_up:     np.ndarray,
        wall_normal: np.ndarray,
        wall_w:      float,
        wall_h:      float,
        dt:          float,
        color_rgba:  tuple,          # (r, g, b, a) floats in [0, 1]
    ) -> None:
        """
        Advance all active particles, detect hits, paint texture, recycle slots.
        """
        if not self._active_slots:
            return

        safe_dt = float(max(0.001, min(dt, 0.05)))
        r, g, b, a = color_rgba

        # Zero hit flags before launch so stale flags don't cause false recycles
        self._hit_np[:] = 0
        self._hit_wp.assign(self._hit_np)

        wp.launch(
            kernel=spray_paint_kernel,
            dim=MAX_PARTICLES,
            inputs=[
                self._pos_wp,
                self._dir_wp,
                self._hit_wp,
                wp.vec3(float(wall_pos[0]),    float(wall_pos[1]),    float(wall_pos[2])),
                wp.vec3(float(wall_normal[0]),  float(wall_normal[1]),  float(wall_normal[2])),
                wp.vec3(float(wall_right[0]),   float(wall_right[1]),   float(wall_right[2])),
                wp.vec3(float(wall_up[0]),      float(wall_up[1]),      float(wall_up[2])),
                float(wall_w),
                float(wall_h),
                self._tex_wp,
                self._tex_w,
                self._tex_h,
                safe_dt,
                float(r), float(g), float(b), float(a),
            ],
        )
        wp.synchronize()

        # Pull results back to CPU
        self._pos_np[:] = self._pos_wp.numpy()
        self._hit_np[:] = self._hit_wp.numpy()

        # --- Recycle hit slots ---
        hit_indices = [
            s for s in list(self._active_slots)
            if self._hit_np[s] == 1
        ]
        if hit_indices:
            for slot in hit_indices:
                self._active_slots.discard(slot)
                self._free_slots.append(slot)
                self._pos_np[slot] = 0.0
                self._dir_np[slot] = 0.0
            # Push zeroed slots back to GPU
            self._pos_wp.assign(self._pos_np)
            self._dir_wp.assign(self._dir_np)

    def update_stage_prim(self) -> None:
        """
        Write active-particle positions to /World/SprayParticles so they are
        visible in the viewport.  Dead slots are never included.
        Runs every other frame to save stage overhead.
        """
        self._stage_frame += 1
        if self._stage_frame % 2 != 0:
            return

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        active_list = list(self._active_slots)
        pts = [
            Gf.Vec3f(
                float(self._pos_np[s, 0]),
                float(self._pos_np[s, 1]),
                float(self._pos_np[s, 2]),
            )
            for s in active_list
        ] if active_list else []

        prim = stage.GetPrimAtPath(PARTICLES_PRIM_PATH)
        if not prim or not prim.IsValid():
            pts_prim = UsdGeom.Points.Define(stage, PARTICLES_PRIM_PATH)
            pts_prim.CreatePointsAttr(pts)
        else:
            UsdGeom.Points(prim).GetPointsAttr().Set(pts)

    def get_tex_buffer(self) -> wp.array:
        return self._tex_wp
