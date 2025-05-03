# PyQMD: Python Quantum Molecular Dynamics

A 2D quark-level nuclear phenomena simulator using GPU acceleration

## Understanding the Simulation Physics

### What You're Seeing

When you run the simulation, you'll see three main types of particles:

1. **Quarks** (Small colored dots)
   - **Up quarks**: Brighter colors (Red, Green, Blue)
   - **Down quarks**: Softer colors (Red, Green, Blue)
   - The colors represent the "color charge" from Quantum Chromodynamics (QCD)

2. **Baryons** (Larger circles)
   - **Protons** (Yellow): Made of 2 up quarks + 1 down quark (uud)
   - **Neutrons** (Gray): Made of 1 up quark + 2 down quarks (udd)
   - **Delta++** (Magenta): Made of 3 up quarks (uuu)
   - **Delta-** (Cyan): Made of 3 down quarks (ddd)

3. **Electrons** (Small yellow dots)
   - Move quickly due to their low mass
   - Attracted to protons, repelled by other electrons

### Physics Behind the Simulation

#### Strong Force (QCD)

- The strong force holds quarks together to form baryons
- Works through "color charge": Red, Green, and Blue
- Quarks with complementary colors attract each other (e.g., Red attracts Green and Blue)
- A baryon must have all three colors to be stable (color neutral)
- Very strong at short distances, rapidly weakens at longer distances

#### Electromagnetic Force

- Works through electric charge
- Like charges repel, opposite charges attract
- Up quarks have +2/3 charge
- Down quarks have -1/3 charge
- Electrons have -1 charge
- Protons have +1 charge (2/3 + 2/3 - 1/3)
- Neutrons have 0 charge (2/3 - 1/3 - 1/3)

#### Baryon Formation

1. Three quarks with complementary colors come close together
2. Strong force binds them into a baryon
3. Their quark flavors (up or down) determine the baryon type:
   - 2 ups + 1 down = proton
   - 1 up + 2 downs = neutron
   - 3 ups = Delta++
   - 3 downs = Delta-

#### Why Some Particles Move More Than Others

- Lighter particles move faster (F = ma, so a = F/m)
- Free quarks can move independently but are affected by forces
- Bound quarks (in baryons) move as a group but still have internal motion
- Electrons are much lighter and move much faster than quarks

#### Vector Forces Acting on Particles

Each particle experiences a combination of these forces:

1. **Strong Force Vectors**:
   - Direction: Toward complementary color quarks, away from same color
   - Magnitude: Very strong at short distances, drops off quickly with distance
   - Only applies between quarks and diminishes beyond ~3 femtometers

2. **Electromagnetic Force Vectors**:
   - Direction: Toward opposite charges, away from like charges
   - Magnitude: Proportional to 1/r²
   - Long range but weaker than the strong force

3. **Boundary Forces**:
   - Direction: Away from simulation boundaries
   - Keeps particles confined within the simulation box

4. **Combined Motion**:
   - Particles accelerate in response to the vector sum of all forces
   - Velocity is updated each time step according to the acceleration
   - Position is updated each time step according to the velocity

#### Why Baryon Types Can Change

In a real nucleus, baryons maintain stable identities. But in our simplified simulation:

1. Weak nuclear force can cause quarks to change flavor (up ↔ down)
2. This can change the baryon type (e.g., proton → neutron)
3. The simulation "locks in" a baryon type after seeing consistent behavior

## Tips for Observing Physics Phenomena

1. **Watch for baryon formation**: When three quarks come together
2. **Notice charge effects**: See how particles with opposite charges attract
3. **Look for clustering**: Baryons tend to group together through residual strong force
4. **Observe electrons**: They orbit around positive charges like electrons in an atom

## Project Structure

- `src/quark.py`: Quark class and helpers
- `src/baryon.py`: Baryon grouping logic
- `src/electron.py`: Electron simulation
- `src/nucleus.py`: Nucleus containers & rules
- `src/physics.py`: Physical constants & CPU utilities
- `src/cuda_kernels.py`: All CUDA kernels
- `src/sim.py`: Main simulation loop & GPU transfers
- `src/visualize.py`: ModernGL GPU-driven renderer

## Dependencies

- Python 3.10+
- NumPy
- Numba (for CUDA)
- ModernGL and GLFW
- SciPy (optional)