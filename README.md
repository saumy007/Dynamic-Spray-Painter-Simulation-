# Dynamic Spray Painter Simulation

A GPU-accelerated spray painting simulator for **NVIDIA Isaac Sim 5.0** that realistically simulates spray paint particles, collision detection, and texture painting in a physics-based simulation environment.

📄 **Detailed Documentation**: [isaac-sim_extension_Dynamic_spray.pdf](https://github.com/user-attachments/files/27106640/isaac-sim_extension_Dynamic_spray.pdf)

---

## 🎯 Overview

This extension adds dynamic spray painting capabilities to NVIDIA Isaac Sim. It allows you to:
- **Paint in real-time** on virtual surfaces using spray particles
- **Control spray parameters** like particle size, emission rate, and spray cone angle
- **Adjust canvas size** dynamically for different surface dimensions
- **Save painted images** as PNG files for post-processing
- **Visualize particles** in real-time with customizable colors and rendering

The simulator uses **NVIDIA Warp** for GPU-accelerated particle physics and collision detection, enabling high-performance simulations with up to 2000 particles.

---

## 📋 Features

### Core Features
- ✅ **Real-time Spray Simulation** - GPU-accelerated particle physics using NVIDIA Warp
- ✅ **Configurable Spray Parameters** - Adjust nozzle size, spray angle, particle velocity, and more
- ✅ **Dynamic Canvas Texture** - Paint directly onto a canvas with real-time texture updates
- ✅ **Particle Visualization** - In-flight particles and frozen impact points rendered in real-time
- ✅ **Collision Detection** - Accurate particle-to-surface collision detection
- ✅ **Paint Color Picker** - RGBA color selection for spray paint
- ✅ **Image Export** - Save painted canvas as PNG files
- ✅ **Customizable Materials** - USD Preview Surface materials for realistic rendering

### Advanced Features
- 🎨 **Material Binding** - Automatic material creation and binding for particles and canvas
- 📊 **Texture Buffer Management** - Efficient GPU-to-CPU texture synchronization
- 🔄 **Round-robin Particle Pool** - Dynamic particle reallocation (10-2000 particles)
- ⚡ **Performance Controls** - Adjustable texture push interval for optimization

---

## 🛠️ Installation

### Prerequisites

Before installing this extension, ensure you have:

1. **NVIDIA Isaac Sim 5.0** or later
   - [Download from NVIDIA](https://developer.nvidia.com/isaac-sim)
   - Supports Linux and Windows

2. **Required Python Packages**
   - `numpy` - Numerical computations
   - `warp` - NVIDIA Warp for GPU computing
   - `PIL` or `opencv-python` - Optional (for PNG image saving)

3. **System Requirements**
   - NVIDIA GPU with CUDA support (RTX series recommended)
   - Minimum 4GB VRAM
   - Ubuntu 20.04+ or Windows 10/11

### Step-by-Step Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/saumy007/Dynamic-Spray-Painter-Simulation-.git
cd Dynamic-Spray-Painter-Simulation-
```

#### Step 2: Locate Your Isaac Sim Extensions Directory
Isaac Sim looks for extensions in multiple locations. The default extension directory is:

**Linux:**
```bash
~/.local/share/ov/pkg/isaac-sim-*/exts/
```

**Windows:**
```bash
%APPDATA%\NVIDIA\isaac-sim\exts\
```

#### Step 3: Create the Extension Structure
Create the required directory structure and copy the extension:

```bash
# Navigate to your Isaac Sim extensions directory
cd ~/.local/share/ov/pkg/isaac-sim-<version>/exts/

# Create the extension directory
mkdir -p company/hello/world1

# Copy the extension files
cp -r /path/to/Dynamic-Spray-Painter-Simulation-/company.hello.world1-1.0.0/* \
      company/hello/world1/
```

Or, create a symbolic link:
```bash
ln -s /path/to/Dynamic-Spray-Painter-Simulation-/company.hello.world1-1.0.0 \
      company/hello/world1
```

#### Step 4: (Optional) Install Python Dependencies
If you haven't already installed the required packages:

```bash
# Using pip with Isaac Sim's Python environment
/path/to/isaac-sim/python.sh -m pip install \
    numpy \
    warp \
    Pillow
```

#### Step 5: Verify Installation
1. Open **NVIDIA Isaac Sim**
2. Navigate to **Window → Extensions**
3. Search for "spray" or "company hello world1"
4. You should see the extension in the list
5. Enable it by checking the checkbox

---

## 🚀 Quick Start

### Loading the Extension

1. **Start Isaac Sim** and open or create a scene
2. **Enable the Extension**:
   - Go to **Window → Extensions**
   - Search for "company hello world1"
   - Check the box to enable it

3. **The Spray Painter UI** will appear on the right side

### Using the Spray Painter

#### 1. **Set Up the Scene**
   - You need a **Canvas Plane** at path `/World/CanvasPlane`
   - You need a **Spray Nozzle** at path `/World/SprayNozzle` (created automatically)
   - Adjust the nozzle position and canvas size as needed

#### 2. **Configure Spray Parameters**

The UI provides several sections:

**🎨 Paint Color**
- Use the color picker to select your spray paint color (RGBA)

**🔧 Nozzle Settings**
- **Radius (m)**: Physical radius of the nozzle cylinder (default: 0.10m)
- **Height (m)**: Height of the nozzle cylinder (default: 0.50m)
- **Cone spread (°)**: Spray cone angle in degrees (default: 5.0°)
  - Larger values = wider spray pattern

**🖼️ Canvas Size (UV mapping)**
- **Canvas width (m)**: Width of your canvas plane (default: 2.0m)
- **Canvas height (m)**: Height of your canvas plane (default: 2.0m)
- **⚠️ Important**: Set these to match your actual `/World/CanvasPlane` dimensions

**💨 Particles**
- **Max particles**: Maximum particles in the pool (default: 1000, range: 10-2000)
- **Emit per tick**: Particles emitted per frame while spraying (default: 20)
- **Speed multiplier**: Particle velocity scale (default: 5.0)
- **Particle display size (m)**: Diameter of particles (default: 0.02m)

**💥 Impact**
- **Splat radius (px)**: Paint splat size in pixels (default: 1)
  - 0 = 1×1 pixel
  - 1 = 3×3 pixels
  - 2 = 5×5 pixels
  - 5 = 11×11 pixels

**⚡ Performance**
- **Texture push interval**: Push texture to viewport every N frames (default: 4)
  - Higher values = better performance but less frequent updates

#### 3. **Spray Paint**
- **Hold the "🎨 Hold to Spray" button** to emit paint particles
- Particles travel from the nozzle toward the canvas
- When particles hit the canvas, they:
  - Stick to the surface (frozen particles)
  - Paint the canvas texture

#### 4. **Save Your Work**
- Click **"💾 Save image"** to export the canvas as a PNG file
- Files are saved in the extension directory with timestamp: `paint_saved_YYYYMMDD_HHMMSS.png`

#### 5. **Reset Canvas**
- Click **"🔄 Reset canvas"** to clear all painted content and reset particles

---

## 📁 Project Structure

```
Dynamic-Spray-Painter-Simulation-/
├── company.hello.world1-1.0.0/
│   ├── company/
│   │   └── hello/
│   │       └── world1/
│   │           ├── __init__.py                 # Package initialization
│   │           ├── extension.py                # Main extension code
│   │           ├── ui.py                       # UI components
│   │           ├── nozzle.py                   # Nozzle handling
│   │           ├── collision.py                # Collision detection
│   │           ├── create_prim.py              # Primitive creation
│   │           └── wall_size.py                # Canvas size utilities
│   ├── config/
│   │   └── extension.toml                      # Extension configuration
│   ├── docs/
│   │   ├── README.md                           # Documentation
│   │   └── CHANGELOG.md                        # Version history
│   └── data/
│       ├── icon.png                            # Extension icon
│       └── preview.png                         # Preview image
├── README.md                                    # This file
└── LICENSE                                      # License information
```

---

## 🎓 How It Works

### Architecture

1. **Extension Framework** (`extension.py`)
   - Manages lifecycle (startup/shutdown)
   - Handles main simulation loop
   - Manages UI and parameter synchronization

2. **GPU Computation** (Warp Kernel)
   - Particle physics simulation
   - Collision detection with canvas plane
   - Texture splatting (painting)
   - Runs on NVIDIA GPU for high performance

3. **Rendering** (USD/Hydra)
   - Particle visualization as UsdGeom.Points
   - Material binding for realistic colors
   - Dynamic texture updates
   - Real-time viewport rendering

4. **User Interface** (Omni UI)
   - Parameter controls via sliders
   - Color picker for paint selection
   - Spray button for emission control
   - Save and reset functionality

### Simulation Pipeline

```
┌─────────────────────────────────────────────┐
│ 1. Read UI Parameters                       │
│    (nozzle size, spray angle, color, etc.)  │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 2. Emit New Particles                       │
│    (from nozzle, in spray cone pattern)     │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 3. GPU Computation (Warp Kernel)            │
│    - Update particle positions              │
│    - Detect canvas collisions               │
│    - Paint texture on impact                │
│    - Recycle hit particles                  │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 4. Update Scene                             │
│    - Create/update Points primitives        │
│    - Bind materials with colors             │
│    - Update frozen particle positions       │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│ 5. Render & Display                         │
│    - Viewport shows particles and texture   │
│    - User sees real-time painting           │
└─────────────────────────────────────────────┘
```

---

## 🔧 Configuration

### Recommended Settings

**For High-Quality Painting:**
```
Max particles:        1500
Emit per tick:        30
Speed multiplier:     5.0
Particle size:        0.02m
Splat radius:         2px
Texture push interval: 2 frames
```

**For High Performance:**
```
Max particles:        500
Emit per tick:        10
Speed multiplier:     3.0
Particle size:        0.01m
Splat radius:         0px
Texture push interval: 8 frames
```

### Adjusting Canvas Size

If your canvas plane is a different size:
1. Measure the width and height of `/World/CanvasPlane` in world units
2. In the UI, adjust **Canvas width** and **Canvas height** to match
3. This ensures proper UV mapping for painting

---

## 🐛 Troubleshooting

### Issue: Extension doesn't appear in Extensions window

**Solution:**
1. Verify the directory structure is correct
2. Check the extension.toml file exists at `/config/extension.toml`
3. Restart Isaac Sim
4. Check Isaac Sim logs for errors: `Window → Toggle Console`

### Issue: Particles not showing in viewport

**Solution:**
1. Ensure you have a valid `/World/CanvasPlane` in your scene
2. Check that the nozzle is positioned above the canvas
3. Verify RTX rendering is enabled (not Path Tracing)
4. Check the particle display size slider (increase if too small)

### Issue: Paint doesn't appear on canvas

**Solution:**
1. Verify canvas dimensions match your plane (`Canvas width` and `Canvas height`)
2. Check that nozzle is pointing at the canvas
3. Increase `Emit per tick` to more particles
4. Reduce `Speed multiplier` so particles have time to reach the canvas
5. Verify the material is bound correctly

### Issue: Low performance/frame rate drops

**Solution:**
1. Reduce `Max particles` (try 500-1000)
2. Reduce `Emit per tick` (try 10-15)
3. Increase `Texture push interval` (try 6-8)
4. Reduce particle display size
5. Lower splat radius

### Issue: PNG export fails

**Solution:**
1. Install PIL: `pip install Pillow`
2. Or install OpenCV: `pip install opencv-python`
3. Check file write permissions in the extension directory

---

## 📚 API & Advanced Usage

### Accessing from Python

```python
# Get the extension instance
from omni.ext import get_ext_manager
ext = get_ext_manager().get_extension("company.hello.world1")

# Access simulation state (if exposed)
ext._running = True   # Start spraying
ext._running = False  # Stop spraying

# Adjust parameters
ext._p_speed = 8.0
ext._p_particle_size = 0.03
```

### Modifying the Code

The extension code is well-commented and organized:

- **`extension.py`**: Main logic, particle simulation, rendering
- **`ui.py`**: UI component definitions
- **`collision.py`**: Collision detection utilities
- **`nozzle.py`**: Nozzle parameters and constraints
- **`wall_size.py`**: Canvas sizing utilities

---

## 📝 License

This project is provided as-is for educational and research purposes.

NVIDIA Isaac Sim is governed by its own license agreement.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Feedback

- **Issue Tracker**: Report bugs and request features via GitHub Issues
- **Documentation**: Refer to [NVIDIA Isaac Sim Docs](https://docs.omniverse.nvidia.com/isaac-sim/latest/)
- **Warp Documentation**: [NVIDIA Warp Docs](https://docs.omniverse.nvidia.com/warp/latest/)

---

## 🎨 Examples & Tips

### Example 1: Paint a Gradient
1. Set nozzle height to match canvas height
2. Increase cone spread to ~15°
3. Move nozzle from left to right while holding spray button
4. Result: Horizontal gradient on canvas

### Example 2: Create Texture
1. Set small particle size (0.005m)
2. High emit rate (50+)
3. Medium splat radius (1-2px)
4. Spray in circular motions
5. Result: Textured paint effect

### Example 3: Fine Detail Work
1. Reduce particle size to minimum (0.001m)
2. Low emit rate (5)
3. Zero splat radius (0px)
4. Slow down speed multiplier (2-3)
5. Result: Precise, fine painting

---

## 📦 Version History

**v1.0.0** (Current)
- ✅ Initial release
- ✅ Full spray painting simulation
- ✅ GPU acceleration with Warp
- ✅ Real-time texture painting
- ✅ PNG export functionality
- ✅ Comprehensive UI controls

---

## 🔮 Future Enhancements

Planned features for future versions:
- [ ] Multi-color layer support
- [ ] Spray brush patterns and stamps
- [ ] Physics-based particle interactions
- [ ] Undo/Redo functionality
- [ ] Canvas animation playback
- [ ] Integration with physics engine
- [ ] Custom particle emitter shapes
- [ ] Material property controls (roughness, metallic, etc.)

---

**Happy painting! 🎨**
