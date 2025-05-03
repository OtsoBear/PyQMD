# Nuclear Decay Simulation

A GPU-accelerated 2D simulation of atomic nuclei structure and radioactive decay, designed to help visualize concepts from the Finnish high school physics curriculum (FY08 "Aine, säteily ja kvantittuminen").

## Features

- Visualization of nuclear structure with protons and neutrons
- Simulation of different types of radioactive decay:
  - Alpha decay (emission of alpha particles)
  - Beta-minus decay (neutron → proton + electron)
  - Beta-plus decay (proton → neutron + positron)
  - Gamma decay (emission of gamma rays)
- GPU acceleration for physics calculations using OpenCL
- Interactive controls for time scaling and isotope selection
- Real-time display of decay statistics

## Requirements

- Python 3.6+
- PyGame
- NumPy
- PyOpenCL
- Matplotlib

## Installation

1. Install the required Python packages:
   ```
   pip install pygame numpy pyopencl matplotlib
   ```

2. Clone or download this repository
3. Run the simulation:
   ```
   python nuclear_sim.py
   ```

## Controls

- **Arrow Up/Down**: Increase/decrease simulation speed
- **Space**: Force a decay event
- **Number keys (1-9)**: Select different isotopes
  - 1: Hydrogen-1
  - 2: Helium-4
  - 3: Carbon-12
  - 4: Carbon-14
  - 5: Iron-56
  - 6: Silver-107
  - 7: Gold-197
  - 8: Lead-208
  - 9: Uranium-238
- **ESC**: Exit the simulation

## Physics Background

This simulation demonstrates:

- Nuclear structure with protons and neutrons
- Strong nuclear force and Coulomb repulsion
- Radioactive decay processes
- Half-life concept and decay probability
- Nuclear stability

The simulation uses simplified models of nuclear forces and decay processes for educational purposes.

## Technical Details

- The simulation uses PyOpenCL to offload physics calculations to the GPU
- If GPU acceleration is unavailable, it falls back to a CPU implementation
- The nuclear structure uses a simplified model of nuclear forces

## Educational Context

This simulation was developed to support the Finnish high school physics curriculum, specifically section FY08 "Aine, säteily ja kvantittuminen" (Matter, Radiation, and Quantization), focusing on:
- Structure and changes of atomic nuclei
- Radioactive decay
