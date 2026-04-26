"""
ui.py
-----
'Dynamic Spray' omni.ui window.

Features
--------
* Hold-to-spray  : the 'Spray' button fires on mouse-press and stops on
                   mouse-release using omni.ui's  set_mouse_pressed_fn  /
                   set_mouse_released_fn  callbacks.
* Color picker   : an omni.ui.ColorWidget lets the user choose RGBA paint
                   color live.  The current (r, g, b, a) tuple is always
                   readable via  SprayUI.get_color().
* Setup Canvas   : one-time button to rewire the existing WallPaintMaterial
                   to the dynamic:// texture provider.
* Save image     : dumps the current canvas to a PNG file.

All business-logic callbacks are injected from extension.py so this module
has zero knowledge of simulation internals.
"""

import omni.ui as ui


class SprayUI:
    """
    Parameters
    ----------
    on_setup_canvas  : callable  — rewire material + create texture provider
    on_spray_start   : callable  — called when Spray button is pressed
    on_spray_stop    : callable  — called when Spray button is released
    on_save_image    : callable  — save canvas PNG
    """

    def __init__(
        self,
        on_setup_canvas,
        on_spray_start,
        on_spray_stop,
        on_save_image,
    ):
        self._on_setup_canvas = on_setup_canvas
        self._on_spray_start  = on_spray_start
        self._on_spray_stop   = on_spray_stop
        self._on_save_image   = on_save_image

        # Default paint color: red, fully opaque
        self._color = [1.0, 0.0, 0.0, 1.0]

        self._window       = None
        self._color_model  = None
        self._build()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_color(self) -> tuple:
        """Returns current (r, g, b, a) as floats in [0, 1]."""
        if self._color_model is not None:
            try:
                items = self._color_model.get_item_children(None)
                self._color = [
                    self._color_model.get_item_value_model(c, 0).as_float
                    for c in items[:4]
                ]
            except Exception:
                pass
        return tuple(self._color)

    def destroy(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build(self):
        self._window = ui.Window("Dynamic Spray", width=340, height=290)
        with self._window.frame:
            with ui.VStack(spacing=8, style={"margin": 6}):

                ui.Label(
                    "Move / Rotate  '/World/SprayNozzle'  in viewport",
                    style={"color": 0xFFAAAAAA, "font_size": 13},
                )

                ui.Button(
                    "Setup Canvas Texture",
                    clicked_fn=self._on_setup_canvas,
                    height=30,
                    tooltip="Rewires WallPaintMaterial to the live dynamic texture provider. Run once.",
                )

                ui.Separator()

                # --- Paint color picker ---
                with ui.HStack(height=28):
                    ui.Label("Paint color:", width=90, style={"font_size": 14})
                    self._color_model = ui.ColorWidget(
                        self._color[0],
                        self._color[1],
                        self._color[2],
                        self._color[3],
                        width=ui.Fraction(1),
                        height=28,
                    ).model

                ui.Separator()

                # --- Hold-to-spray button ---
                spray_btn = ui.Button(
                    "🎨  Hold to Spray",
                    height=44,
                    style={
                        "font_size": 16,
                        "background_color": 0xFF2255AA,
                        "color": 0xFFFFFFFF,
                    },
                )
                # Press = start, Release = stop
                spray_btn.set_mouse_pressed_fn(
                    lambda x, y, btn, mod: self._on_spray_start()
                )
                spray_btn.set_mouse_released_fn(
                    lambda x, y, btn, mod: self._on_spray_stop()
                )

                ui.Separator()

                ui.Button(
                    "💾  Save paint image",
                    clicked_fn=self._on_save_image,
                    height=30,
                )
