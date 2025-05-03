"""
Main simulation class and logic.
"""
import numpy as np
import time
import pickle
from numba import cuda
import warnings
import sys

# Import physics constants
from physics import (
    binding_thresh, strong_const, coulomb_const, 
    dt, weak_prob, gravity_const, electron_mass,
    DEBUG_FORCES, DEBUG_MOVEMENT
)
from electron import Electron, electrons_to_array

# Try to import cuda random module (may not be available in all Numba versions)
try:
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
    HAS_CUDA_RANDOM = True
except ImportError:
    warnings.warn("Numba CUDA random module not available. Using NumPy random instead.")
    HAS_CUDA_RANDOM = False

# Import our modules
from quark import Quark, QuarkFlavor, QuarkColor, create_quark_dtype, quarks_to_array
from baryon import Baryon, BaryonType, create_baryon_dtype, baryons_to_array
from nucleus import Nucleus, create_nucleus_dtype, nuclei_to_array

# Import CUDA kernels
from cuda_kernels import (
    compute_forces_kernel,
    integrate_kernel,
    group_quarks_kernel,
    decay_step_kernel,
    collision_reactions_kernel,
    fission_kernel
)

class Simulation:
    """Main simulation class handling all aspects of the QMD simulation."""
    
    def __init__(self, num_particles=1000, box_size=100.0, dt=dt):
        """
        Initialize the simulation.
        
        Args:
            num_particles: Number of initial quarks
            box_size: Size of the simulation box in fm
            dt: Time step in seconds
        """
        self.num_particles = num_particles
        self.box_size = box_size
        self.dt = dt
        
        # Initialize simulation time
        self.time = 0.0
        self.steps = 0
        
        # Initialize empty collections
        self.quarks = []
        self.baryons = []
        self.nuclei = []
        self.electrons = []  # Add electrons list
        
        # Generate initial particles
        self._generate_initial_quarks()
        self._generate_initial_electrons(num_electrons=num_particles//6)  # ~1 electron per 6 quarks (reasonable ratio)
        
        # Prepare GPU buffers
        self._initialize_gpu_buffers()
        
        print(f"Simulation initialized with {num_particles} quarks")
    
    def _generate_initial_quarks(self):
        """Generate initial quark distribution with more realistic clustering."""
        self.quarks = []
        
        # Create quarks in a realistic arrangement - concentrated in the center
        center = self.box_size / 2
        
        # Create nucleon-like clusters with correct quark combinations
        num_nucleons = self.num_particles // 3
        particles_left = self.num_particles
        
        # Create a more concentrated initial distribution
        # This helps form baryons more naturally
        cluster_radius = self.box_size * 0.2  # Tighter clustering to encourage interactions
        
        for cluster in range(num_nucleons):
            # Determine if this is a proton (uud) or neutron (udd)
            is_proton = cluster % 2 == 0
            
            # Position for this nucleon cluster - more centrally concentrated
            angle = 2 * np.pi * cluster / num_nucleons
            # Gaussian distance distribution from center (more realistic nuclear distribution)
            distance = np.random.normal(0, cluster_radius/2)
            distance = min(cluster_radius, abs(distance))  # Constrain to cluster radius
            
            cluster_x = center + distance * np.cos(angle)
            cluster_y = center + distance * np.sin(angle)
            
            # Create a tight group of 3 quarks for this nucleon - closer together
            for i in range(3):
                if particles_left <= 0:
                    break
                    
                # Very small offset to keep quarks close together
                offset_angle = 2 * np.pi * i / 3 + np.random.uniform(-0.05, 0.05)  # Tighter
                offset_distance = 0.5 + np.random.uniform(-0.1, 0.1)  # Smaller offsets
                
                x = cluster_x + offset_distance * np.cos(offset_angle)
                y = cluster_y + offset_distance * np.sin(offset_angle)
                
                # Constrain positions to box
                x = max(1.0, min(self.box_size - 1.0, x))
                y = max(1.0, min(self.box_size - 1.0, y))
                
                # Initial velocity - smaller to encourage binding
                vel_magnitude = np.random.uniform(0.01, 0.1)  # Much lower velocity
                orbital_angle = offset_angle + np.pi/2  # Perpendicular to radius for orbital motion
                vx = vel_magnitude * np.cos(orbital_angle)
                vy = vel_magnitude * np.sin(orbital_angle)
                
                # For proton: uud quarks, for neutron: udd quarks
                # Ensure specific combinations that match real particles
                if is_proton:
                    if i < 2:  # Two up quarks for proton
                        flavor = QuarkFlavor.UP
                    else:
                        flavor = QuarkFlavor.DOWN
                else:
                    if i < 1:  # One up quark for neutron
                        flavor = QuarkFlavor.UP
                    else:
                        flavor = QuarkFlavor.DOWN
                
                # Ensure color neutrality (one of each color)
                color = QuarkColor(i % 3)
                
                # Create quark
                quark = Quark(x, y, vx, vy, flavor, color)
                self.quarks.append(quark)
                particles_left -= 1
        
        print(f"Generated {self.num_particles} quarks in {num_nucleons} realistic nucleon clusters")
    
    def _generate_initial_electrons(self, num_electrons):
        """Generate initial electron distribution."""
        self.electrons = []
        
        for i in range(num_electrons):
            # Random position - electrons should be more spread out than quarks
            x = np.random.uniform(0.1 * self.box_size, 0.9 * self.box_size)
            y = np.random.uniform(0.1 * self.box_size, 0.9 * self.box_size)
            
            # Electrons move faster than quarks (lighter)
            vx = np.random.normal(0, 0.8)  # Higher velocity
            vy = np.random.normal(0, 0.8)
            
            # Create electron
            electron = Electron(x, y, vx, vy)
            self.electrons.append(electron)
        
        print(f"Generated {num_electrons} electrons")
    
    def _initialize_gpu_buffers(self):
        """Initialize all GPU buffers needed for the simulation."""
        # Convert quarks to numpy arrays
        self.quark_array = quarks_to_array(self.quarks)
        
        # Allocate GPU arrays
        self.d_quarks = cuda.to_device(self.quark_array)
        self.d_forces_x = cuda.to_device(np.zeros(len(self.quarks), dtype=np.float32))
        self.d_forces_y = cuda.to_device(np.zeros(len(self.quarks), dtype=np.float32))
        
        # Baryon detection arrays - INCREASE BUFFER SIZE to prevent overflow warnings
        # Update: Increase to handle 2000 candidates
        max_baryons = max(2000, len(self.quarks) * len(self.quarks) // 2)
        self.d_baryon_candidates = cuda.to_device(np.zeros((max_baryons, 3), dtype=np.int32))
        self.d_baryon_counters = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Decay event arrays
        max_decays = len(self.quarks) // 10  # Assume ~10% of quarks decay per step
        self.d_decay_events = cuda.to_device(np.zeros((max_decays, 2), dtype=np.int32))
        self.d_decay_counters = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Reaction event arrays
        max_reactions = max_baryons // 10  # Assume ~10% of baryons react per step
        self.d_reaction_events = cuda.to_device(np.zeros((max_reactions, 3), dtype=np.float32))
        self.d_reaction_counters = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Fission event arrays
        max_fissions = 100  # Maximum number of fission events per step
        self.d_fission_events = cuda.to_device(np.zeros(max_fissions, dtype=np.int32))
        self.d_fission_counters = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Initialize CUDA random number generator or fallback to NumPy
        if HAS_CUDA_RANDOM:
            self.rng_states = create_xoroshiro128p_states(1024, seed=int(time.time()))
        else:
            # Use NumPy random instead as a fallback
            np.random.seed(int(time.time()))
            print("Using NumPy random as fallback for CUDA random")
        
        # Set up grid and block sizes
        self.threads_per_block = 256
        self.blocks_per_grid = (len(self.quarks) + self.threads_per_block - 1) // self.threads_per_block
    
    def step(self):
        """Advance simulation by one time step."""
        # Reset counters
        cuda.to_device(np.zeros(1, dtype=np.int32), to=self.d_baryon_counters)
        cuda.to_device(np.zeros(1, dtype=np.int32), to=self.d_decay_counters)
        cuda.to_device(np.zeros(1, dtype=np.int32), to=self.d_reaction_counters)
        cuda.to_device(np.zeros(1, dtype=np.int32), to=self.d_fission_counters)
        
        # Run kernels
        self._run_force_calculation()
        self._run_integration()
        
        # Run baryon detection more frequently
        if self.steps % 5 == 0:
            self._run_baryon_detection()
            self._run_decay_simulation()
        
        # If we have baryons, run nuclear reactions and fission
        if len(self.baryons) > 0:
            self._run_reaction_simulation()
            self._run_fission_simulation()
        
        # Update electrons
        self._update_electrons()
        
        # Update time
        self.time += self.dt
        self.steps += 1
        
        # Update CPU objects every step
        self._update_cpu_objects()
        
        # Print debug info
        if DEBUG_MOVEMENT and self.steps % 10 == 0:
            self._print_movement_debug()
    
    def _run_force_calculation(self):
        """Run the force calculation kernel."""
        compute_forces_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_quarks, self.d_forces_x, self.d_forces_y
        )
    
    def _run_integration(self):
        """Run the integration kernel."""
        integrate_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_quarks, self.d_forces_x, self.d_forces_y
        )
    
    def _run_baryon_detection(self):
        """Run the baryon detection kernel with improved stability."""
        group_quarks_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_quarks, self.d_baryon_candidates, self.d_baryon_counters
        )
        
        # Process baryon candidates on CPU
        baryon_counters = self.d_baryon_counters.copy_to_host()
        num_candidates = baryon_counters[0]
        
        if num_candidates > 0:
            # Check if we're within bounds
            max_candidates = self.d_baryon_candidates.shape[0]
            if num_candidates > max_candidates:
                print(f"Warning: {num_candidates} baryon candidates found, but buffer only holds {max_candidates}")
                num_candidates = max_candidates
                
            baryon_candidates = self.d_baryon_candidates.copy_to_host()[:num_candidates]
            
            # Process each candidate
            new_baryons_formed = 0
            for i in range(num_candidates):
                quark_indices = baryon_candidates[i]
                
                # Check if any of these quarks are already in a baryon
                skip = False
                for baryon in self.baryons:
                    if any(idx in baryon.quark_indices for idx in quark_indices):
                        skip = True
                        break
                
                if not skip:
                    # Create a new baryon
                    try:
                        baryon = Baryon(quark_indices, self.quark_array)
                        self.baryons.append(baryon)
                        new_baryons_formed += 1
                        
                        # Mark these quarks as bound
                        for idx in quark_indices:
                            self.quark_array[idx]['is_bound'] = 1
                        
                        # Copy updated quark array back to device
                        cuda.to_device(self.quark_array, to=self.d_quarks)
                        
                    except (ValueError, IndexError) as e:
                        # Skip invalid baryon configurations
                        continue

            if new_baryons_formed > 0:
                print(f"Formed {new_baryons_formed} new baryons! Total: {len(self.baryons)}")
    
    def _run_decay_simulation(self):
        """Run the weak decay simulation kernel."""
        decay_step_kernel[self.blocks_per_grid, self.threads_per_block](
            self.d_quarks, self.rng_states, self.d_decay_events, self.d_decay_counters
        )
        
        # Process decay events on CPU every few steps
        if self.steps % 10 == 0:
            decay_counters = self.d_decay_counters.copy_to_host()
            num_decays = decay_counters[0]
            
            if num_decays > 0:
                # Copy decay events to CPU
                decay_events = self.d_decay_events.copy_to_host()[:num_decays]
                
                # Copy quark data back to CPU
                self.quark_array = self.d_quarks.copy_to_host()
                
                # Process each decay event (simplified model)
                for i in range(num_decays):
                    quark_idx = decay_events[i, 0]
                    decay_type = decay_events[i, 1]
                    
                    # Update the quark object
                    quark = self.quarks[quark_idx]
                    quark.flavor = QuarkFlavor(self.quark_array[quark_idx]['flavor'])
                    quark.charge = self.quark_array[quark_idx]['charge']
                    quark.mass = self.quark_array[quark_idx]['mass']
                    
                    # Note: In a more complete simulation, we would emit leptons here
                
                # Update baryons that contain these quarks
                affected_baryons = []
                for baryon in self.baryons:
                    for quark_idx in range(num_decays):
                        if decay_events[quark_idx, 0] in baryon.quark_indices:
                            affected_baryons.append(baryon)
                            break
                
                for baryon in affected_baryons:
                    baryon.update_from_quarks(self.quark_array)
    
    def _run_reaction_simulation(self):
        """Run nuclear reaction simulation."""
        # In a full implementation, this would handle baryon-baryon reactions
        pass
    
    def _run_fission_simulation(self):
        """Run nuclear fission simulation."""
        # In a full implementation, this would handle fission of large nuclei
        pass
    
    def _update_electrons(self):
        """Update electron positions and velocities based on forces."""
        # Create temporary arrays for quarks and baryons
        quark_pos = np.array([[q.x, q.y] for q in self.quarks])
        quark_charge = np.array([q.charge for q in self.quarks])
        baryon_pos = np.array([[b.x, b.y] for b in self.baryons]) if self.baryons else np.empty((0, 2))
        baryon_charge = np.array([1.0 if b.type == BaryonType.PROTON or b.type == BaryonType.DELTA_PLUS_PLUS 
                                 else (-1.0 if b.type == BaryonType.DELTA_MINUS else 0.0) 
                                 for b in self.baryons]) if self.baryons else np.empty(0)
        
        # Update each electron with GUARANTEED visible motion
        for electron in self.electrons:
            # Add large forced motion term
            fx = 100.0 * np.sin(electron.x * 5.0 + self.time * 10)
            fy = 100.0 * np.cos(electron.y * 5.0 + self.time * 10)
            
            # Apply electromagnetic force from all quarks
            for i, (pos, charge) in enumerate(zip(quark_pos, quark_charge)):
                dx = pos[0] - electron.x
                dy = pos[1] - electron.y
                r_squared = dx*dx + dy*dy + 0.01  # Add small constant to prevent extreme forces
                r = np.sqrt(r_squared)
                
                # Electrons feel stronger forces
                force_magnitude = coulomb_const * 10.0 * electron.charge * charge / r_squared
                
                fx += force_magnitude * dx / r
                fy += force_magnitude * dy / r
            
            # Apply electromagnetic force from all baryons
            for i, (pos, charge) in enumerate(zip(baryon_pos, baryon_charge)):
                dx = pos[0] - electron.x
                dy = pos[1] - electron.y
                r_squared = dx*dx + dy*dy + 0.01
                r = np.sqrt(r_squared)
                
                force_magnitude = coulomb_const * 20.0 * electron.charge * charge / r_squared
                
                fx += force_magnitude * dx / r
                fy += force_magnitude * dy / r
            
            # Apply boundary forces
            box_margin = 1.0
            boundary_strength = 200.0
            
            if electron.x < box_margin:
                fx += boundary_strength * (box_margin - electron.x)
            elif electron.x > self.box_size - box_margin:
                fx -= boundary_strength * (electron.x - (self.box_size - box_margin))
                
            if electron.y < box_margin:
                fy += boundary_strength * (box_margin - electron.y)
            elif electron.y > self.box_size - box_margin:
                fy -= boundary_strength * (electron.y - (self.box_size - box_margin))
            
            # Calculate acceleration
            ax = fx / electron.mass
            ay = fy / electron.mass
            
            # Massive acceleration boost
            accel_boost = 5000.0  # Extreme boost
            ax *= accel_boost
            ay *= accel_boost
            
            # Update velocity with minimal damping
            electron.vx = 0.99 * electron.vx + ax * self.dt
            electron.vy = 0.99 * electron.vy + ay * self.dt
            
            # EXTREME movement scale for visibility
            movement_scale = 10000.0  # Massive amplification
            dx = electron.vx * self.dt * movement_scale
            dy = electron.vy * self.dt * movement_scale
            
            # Always ensure visible movement
            min_move = 0.1  # Large enough to be visible
            random_angle = np.random.random() * 2 * np.pi
            dx = max(min_move * np.cos(random_angle), abs(dx)) * np.sign(dx or 1)
            dy = max(min_move * np.sin(random_angle), abs(dy)) * np.sign(dy or 1)
            
            # Apply position change
            electron.x += dx
            electron.y += dy
            
            # Keep electron in box but allow them to move around more
            electron.x = max(0.1, min(self.box_size - 0.1, electron.x))
            electron.y = max(0.1, min(self.box_size - 0.1, electron.y))
    
    def _print_movement_debug(self):
        """Print debug information about particle movement."""
        # Sample a few particles to see their velocities and positions
        print(f"\n--- Movement Debug at step {self.steps} ---")
        
        # Check quarks with more precision
        if self.quarks:
            print("QUARK MOVEMENT:")
            for i in range(min(5, len(self.quarks))):
                q = self.quarks[i]
                print(f"  Quark {i}: pos=({q.x:.6f}, {q.y:.6f}), vel=({q.vx:.6f}, {q.vy:.6f}), bound={q.is_bound}")
        
        # Check baryons if any exist
        if self.baryons:
            print("BARYON MOVEMENT:")
            for i in range(min(3, len(self.baryons))):
                b = self.baryons[i]
                print(f"  Baryon {i}: pos=({b.x:.6f}, {b.y:.6f}), type={b.type}")
        
        # Check electrons
        if self.electrons:
            print("ELECTRON MOVEMENT:")
            for i in range(min(3, len(self.electrons))):
                e = self.electrons[i]
                print(f"  Electron {i}: pos=({e.x:.6f}, {e.y:.6f}), vel=({e.vx:.6f}, {e.vy:.6f})")
        
        # Check forces on a sample quark
        if self.quarks and DEBUG_FORCES:
            i = 0  # Sample first quark
            forces = self.d_forces_x.copy_to_host(), self.d_forces_y.copy_to_host()
            print(f"FORCE on Quark {i}: ({forces[0][i]:.3f}, {forces[1][i]:.3f})")
        
        # Print integration parameters
        print(f"DT={dt}, Mass(up)={self.quarks[0].mass if self.quarks else 'N/A'}")
        sys.stdout.flush()  # Ensure output is displayed immediately
    
    def _update_cpu_objects(self):
        """Update CPU-side objects from GPU data."""
        # Copy quark data back to CPU
        self.quark_array = self.d_quarks.copy_to_host()
        
        # Update all quarks to ensure movement
        for i, quark in enumerate(self.quarks):
            quark.x = self.quark_array[i]['x']
            quark.y = self.quark_array[i]['y']
            quark.vx = self.quark_array[i]['vx']
            quark.vy = self.quark_array[i]['vy']
        
        # Update baryons based on constituent quarks' movement
        for baryon in self.baryons:
            baryon.update_from_quarks(self.quark_array)
    
    def get_quark_data(self):
        """Get quark position and property data for visualization."""
        # Ensure data is up to date
        self._update_cpu_objects()
        
        # Extract positions and properties
        positions = np.array([[quark.x, quark.y] for quark in self.quarks])
        flavors = np.array([quark.flavor for quark in self.quarks])
        colors = np.array([quark.color for quark in self.quarks])
        
        return positions, flavors, colors
    
    def get_electron_data(self):
        """Get electron position data for visualization."""
        positions = np.array([[e.x, e.y] for e in self.electrons])
        velocities = np.array([[e.vx, e.vy] for e in self.electrons])
        
        return positions, velocities
    
    def get_baryon_data(self):
        """Get baryon position and property data for visualization."""
        # Extract positions and properties
        positions = np.array([[baryon.x, baryon.y] for baryon in self.baryons])
        types = np.array([baryon.type for baryon in self.baryons])
        
        return positions, types
    
    def save_state(self, filename):
        """
        Save simulation state to file.
        
        Args:
            filename: Path to save the state
        """
        # Ensure all data is up to date
        self._update_cpu_objects()
        
        # Create a state dictionary
        state = {
            'time': self.time,
            'steps': self.steps,
            'quarks': self.quarks,
            'baryons': self.baryons,
            'nuclei': self.nuclei
        }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
        print(f"Simulation state saved to {filename}")
    
    def load_state(self, filename):
        """
        Load simulation state from file.
        
        Args:
            filename: Path to the state file
        """
        # Load from file
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Restore state
        self.time = state['time']
        self.steps = state['steps']
        self.quarks = state['quarks']
        self.baryons = state['baryons']
        self.nuclei = state['nuclei']
        
        # Re-initialize GPU buffers
        self._initialize_gpu_buffers()
        
        print(f"Simulation state loaded from {filename}")
