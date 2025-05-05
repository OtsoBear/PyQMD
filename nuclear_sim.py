import pygame
import numpy as np
import pyopencl as cl
import pyopencl.array
import random
import os
from enum import Enum
import math
import time
from collections import deque
import matplotlib.pyplot as plt
import sys
import subprocess
import logging
import threading
import ctypes

# Set environment variable for OpenCL compiler output
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NuclearSim")

# Try to install siphash24 to fix the hash warning
try:
    import siphash24
except ImportError:
    logger.info("siphash24 not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "siphash24"])
        logger.info("siphash24 installed successfully")
    except Exception as e:
        logger.warning(f"Failed to install siphash24: {e}")

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
SIMULATION_WIDTH = 800
SIMULATION_HEIGHT = 800
BACKGROUND_COLOR = (0, 0, 0)
PROTON_COLOR = (255, 100, 100)
NEUTRON_COLOR = (100, 100, 255)
GAMMA_COLOR = (0, 255, 0)
ELECTRON_COLOR = (0, 255, 255)
ALPHA_COLOR = (255, 200, 0)
FPS = 60

# Physics constants (in simulation units)
# These values are proportionally correct but scaled for simulation
STRONG_FORCE_STRENGTH = 150.0      # Strong nuclear force 
COULOMB_STRENGTH = 30.0           # Electromagnetic repulsion
PAULI_STRENGTH = 35.0             # Pauli exclusion principle repulsion
GRAVITY_STRENGTH = 0.01           # Gravitational attraction (very weak)
WEAK_FORCE_STRENGTH = 1.0         # Weak nuclear force

# Force ranges
STRONG_FORCE_RANGE = 10.0         # ~1.5 femtometer (scaled)
PAULI_EXCLUSION_RANGE = 8.0       # Range of Pauli exclusion interaction

# Scale conversion for accurate visualization
FM_PER_UNIT = 0.5                # Each simulation unit = 0.5 femtometers
NUCLEON_RADIUS_FM = 0.9          # Physical nucleon radius in femtometers
NUCLEON_RADIUS = 2.5             # Nucleon radius in simulation units (~1.25 fm)

# Physics constants for time scales
PLANCK_TIME = 5.39e-44  # Planck time in seconds
ATTOSECOND = 1e-18      # Attosecond in seconds 
FEMTOSECOND = 1e-15     # Femtosecond in seconds
PICOSECOND = 1e-12      # Picosecond in seconds
NANOSECOND = 1e-9       # Nanosecond in seconds
SECOND = 1.0            # Standard second
MINUTE = 60.0           # Minute in seconds
HOUR = 3600.0           # Hour in seconds
YEAR = 31557600.0       # Year in seconds (365.25 days)

# Particle types
class ParticleType(Enum):
    PROTON = 0
    NEUTRON = 1
    ALPHA = 2
    ELECTRON = 3
    GAMMA = 4

# Decay types
class DecayType(Enum):
    NONE = 0
    ALPHA = 1
    BETA_MINUS = 2
    BETA_PLUS = 3
    GAMMA = 4

class Particle:
    def __init__(self, x, y, particle_type, vx=0, vy=0):
        self.x = x
        self.y = y
        self.type = particle_type
        self.vx = vx
        self.vy = vy
        
        # Physical radius in simulation units (nucleons have ~0.8-0.9 fm radius)
        if particle_type in [ParticleType.PROTON, ParticleType.NEUTRON]:
            self.radius = 2.5  # Physical radius for collision detection
        else:
            self.radius = 1.0  # Smaller radius for decay products
            
        self.lifetime = float('inf')
        if particle_type == ParticleType.ALPHA:
            self.lifetime = 2.0
        elif particle_type == ParticleType.ELECTRON:
            self.lifetime = 3.0
        elif particle_type == ParticleType.GAMMA:
            self.lifetime = 1.0
        self.age = 0
        
    def get_color(self):
        if self.type == ParticleType.PROTON:
            return PROTON_COLOR
        elif self.type == ParticleType.NEUTRON:
            return NEUTRON_COLOR
        elif self.type == ParticleType.ALPHA:
            return ALPHA_COLOR
        elif self.type == ParticleType.ELECTRON:
            return ELECTRON_COLOR
        elif self.type == ParticleType.GAMMA:
            return GAMMA_COLOR
        return (255, 255, 255)

class Nucleus:
    def __init__(self, protons, neutrons, x, y):
        self.protons = protons
        self.neutrons = neutrons
        self.x = x
        self.y = y
        self.particles = []
        self.decay_history = []
        self.initialize_particles()
        self.stability = self.calculate_stability()
        self.decay_time = 0
        self.half_life = self.calculate_half_life()
        
    def initialize_particles(self):
        """Place particles in a physically realistic nuclear arrangement"""
        total = self.protons + self.neutrons
        
        # Nuclear radius follows r = r0 * A^(1/3) formula
        # r0 is approximately 1.2 fm (scaled to our units)
        r0 = 1.2
        nuclear_radius = r0 * (total ** (1/3))
        
        # For more realistic density, use smaller initial radius
        initial_radius = nuclear_radius * 0.7
        
        self.particles = []
        
        # Place particles in shells based on quantum mechanics principles
        # Shell model of nucleus - particles fill shells from inside out
        
        # Calculate approximate number of particles per shell
        shell_capacity = [2, 8, 20, 28, 50, 82, 126]
        shell_radii = [initial_radius * (i+1)/len(shell_capacity) for i in range(len(shell_capacity))]
        
        # Distribute protons and neutrons
        placed_protons = 0
        placed_neutrons = 0
        
        # Helper to place a particle in a shell with some randomization
        def place_in_shell(shell_index, is_proton):
            nonlocal placed_protons, placed_neutrons
            
            # Get shell radius
            shell_radius = shell_radii[min(shell_index, len(shell_radii)-1)]
            
            # Add some randomness to radius (80-100% of shell radius)
            radius = shell_radius * (0.8 + 0.2 * random.random())
            
            # Random angles with attempt to maintain distance from other same-type particles
            max_attempts = 20
            best_angle = 0
            max_min_dist = 0
            
            for _ in range(max_attempts):
                angle = random.uniform(0, 2 * math.pi)
                x = self.x + radius * math.cos(angle)
                y = self.y + radius * math.sin(angle)
                
                # Check distance to other particles of same type
                min_dist = float('inf')
                for p in self.particles:
                    if (p.type == ParticleType.PROTON and is_proton) or \
                       (p.type == ParticleType.NEUTRON and not is_proton):
                        dist = math.sqrt((p.x - x)**2 + (p.y - y)**2)
                        min_dist = min(min_dist, dist)
                
                # If no particles yet or found a better position
                if min_dist == float('inf') or min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_angle = angle
            
            # Use the best position found
            x = self.x + radius * math.cos(best_angle)
            y = self.y + radius * math.sin(best_angle)
            
            if is_proton:
                self.particles.append(Particle(x, y, ParticleType.PROTON))
                placed_protons += 1
            else:
                self.particles.append(Particle(x, y, ParticleType.NEUTRON))
                placed_neutrons += 1
        
        # Distribute particles in shells (simplified nuclear shell model)
        # First place protons and neutrons in pairs for stability
        shell_index = 0
        while placed_protons < self.protons and placed_neutrons < self.neutrons:
            # Place as many pairs as can fit in current shell
            shell_size = shell_capacity[min(shell_index, len(shell_capacity)-1)]
            pairs_to_place = min(shell_size // 2, 
                                min(self.protons - placed_protons, self.neutrons - placed_neutrons))
            
            for _ in range(pairs_to_place):
                place_in_shell(shell_index, True)  # Proton
                place_in_shell(shell_index, False)  # Neutron
            
            shell_index += 1
            if shell_index >= len(shell_capacity):
                shell_index = len(shell_capacity) - 1
        
        # Place any remaining protons or neutrons
        while placed_protons < self.protons:
            place_in_shell(shell_index, True)
        
        while placed_neutrons < self.neutrons:
            place_in_shell(shell_index, False)
    
    def calculate_stability(self):
        """Calculate stability of the nucleus based on N/Z ratio"""
        z = self.protons
        n = self.neutrons
        if z == 0:
            return 0
        
        # Optimal N/Z ratio approximation
        optimal_ratio = 1.0 if z <= 20 else 1 + 0.0075 * z**1.3
        actual_ratio = n / z
        
        # Penalize being too far from optimal ratio
        stability = 1.0 - min(abs(actual_ratio - optimal_ratio) / 2.0, 0.5)
        
        # Magic numbers increase stability
        magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        if z in magic_numbers:
            stability += 0.1
        if n in magic_numbers:
            stability += 0.1
            
        # Normalize to [0,1]
        return max(0, min(stability, 1))
    
    def calculate_half_life(self):
        """Calculate half-life based on stability"""
        if self.stability >= 0.9:
            return float('inf')  # Stable
        elif self.stability >= 0.8:
            return random.uniform(1e9, 1e11)  # Years
        elif self.stability >= 0.6:
            return random.uniform(1e6, 1e9)   # Years
        elif self.stability >= 0.4:
            return random.uniform(1e3, 1e6)   # Years
        elif self.stability >= 0.2:
            return random.uniform(1, 1e3)     # Years
        else:
            return random.uniform(0.01, 1)    # Years
    
    def should_decay(self, dt):
        """Determine if nucleus should decay based on half-life"""
        # Convert half life to seconds for simulation (use scaled time)
        # For simulation, 1 year = 1 second
        scaled_half_life = self.half_life
        if scaled_half_life == float('inf'):
            return False
            
        probability = 1 - 0.5 ** (dt / scaled_half_life)
        return random.random() < probability
    
    def determine_decay_type(self):
        """Determine what type of decay occurs based on nucleus properties"""
        z = self.protons
        n = self.neutrons
        a = z + n
        
        # For heavy nuclei, prefer alpha decay
        if a > 210:
            return DecayType.ALPHA
            
        # For neutron-rich nuclei, prefer beta minus decay
        if n > 1.5 * z:
            return DecayType.BETA_MINUS
            
        # For proton-rich nuclei, prefer beta plus decay
        if z > 1.1 * n:
            return DecayType.BETA_PLUS
            
        # Otherwise random between alpha and beta
        r = random.random()
        if r < 0.4:
            return DecayType.ALPHA
        elif r < 0.7:
            return DecayType.BETA_MINUS
        elif r < 0.9:
            return DecayType.BETA_PLUS
        else:
            return DecayType.GAMMA
    
    def decay(self):
        """Perform radioactive decay"""
        decay_type = self.determine_decay_type()
        self.decay_history.append(decay_type)
        
        if decay_type == DecayType.ALPHA:
            # Remove 2 protons and 2 neutrons, emit alpha particle
            self.protons -= 2
            self.neutrons -= 2
            
            # Find particles to remove (2 protons, 2 neutrons)
            protons_removed = 0
            neutrons_removed = 0
            particles_to_remove = []
            
            # Select particles to remove
            for i, particle in enumerate(self.particles):
                if protons_removed < 2 and particle.type == ParticleType.PROTON:
                    particles_to_remove.append(i)
                    protons_removed += 1
                elif neutrons_removed < 2 and particle.type == ParticleType.NEUTRON:
                    particles_to_remove.append(i)
                    neutrons_removed += 1
                    
                if protons_removed == 2 and neutrons_removed == 2:
                    break
                    
            # Remove particles in reverse order to maintain indices
            particles_to_remove.sort(reverse=True)
            removed_particles = []
            for i in particles_to_remove:
                removed_particles.append(self.particles.pop(i))
            
            # Calculate average position of removed particles
            avg_x = sum(p.x for p in removed_particles) / len(removed_particles)
            avg_y = sum(p.y for p in removed_particles) / len(removed_particles)
            
            # Create alpha particle with velocity away from nucleus
            dx = avg_x - self.x
            dy = avg_y - self.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 1e-6:
                # If at center, choose random direction
                angle = random.uniform(0, 2 * math.pi)
                dx, dy = math.cos(angle), math.sin(angle)
                
            # Normalize and set speed
            speed = 100
            vx = speed * dx / (dist if dist > 0 else 1)
            vy = speed * dy / (dist if dist > 0 else 1)
            
            # Update center of mass
            self.update_center_of_mass()
            
            return [Particle(avg_x, avg_y, ParticleType.ALPHA, vx, vy)]
            
        elif decay_type == DecayType.BETA_MINUS:
            # A neutron decays to a proton and emits an electron
            self.protons += 1
            self.neutrons -= 1
            
            # Find a neutron to convert
            for particle in self.particles:
                if particle.type == ParticleType.NEUTRON:
                    particle.type = ParticleType.PROTON
                    
                    # Random direction for electron
                    angle = random.uniform(0, 2 * math.pi)
                    speed = 150
                    vx = speed * math.cos(angle)
                    vy = speed * math.sin(angle)
                    
                    # Update center of mass
                    self.update_center_of_mass()
                    
                    return [Particle(particle.x, particle.y, ParticleType.ELECTRON, vx, vy)]
            
        elif decay_type == DecayType.BETA_PLUS:
            # A proton decays to a neutron and emits a positron (we'll use electron for simplicity)
            self.protons -= 1
            self.neutrons += 1
            
            # Find a proton to convert
            for particle in self.particles:
                if particle.type == ParticleType.PROTON:
                    particle.type = ParticleType.NEUTRON
                    
                    # Random direction for "positron"
                    angle = random.uniform(0, 2 * math.pi)
                    speed = 150
                    vx = speed * math.cos(angle)
                    vy = speed * math.sin(angle)
                    
                    # Update center of mass
                    self.update_center_of_mass()
                    
                    return [Particle(particle.x, particle.y, ParticleType.ELECTRON, vx, vy)]
                    
        elif decay_type == DecayType.GAMMA:
            # Emit gamma ray without changing nucleus composition
            angle = random.uniform(0, 2 * math.pi)
            speed = 200
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            
            return [Particle(self.x, self.y, ParticleType.GAMMA, vx, vy)]
        
        return []

    def update_particles(self):
        """Update nuclear structure based on current proton/neutron count"""
        # Count current particles by type
        proton_count = sum(1 for p in self.particles if p.type == ParticleType.PROTON)
        neutron_count = sum(1 for p in self.particles if p.type == ParticleType.NEUTRON)
        
        # Calculate center of mass
        if len(self.particles) > 0:
            com_x = sum(p.x for p in self.particles) / len(self.particles)
            com_y = sum(p.y for p in self.particles) / len(self.particles)
            self.x = com_x
            self.y = com_y
        
        # If we have any discrepancies in particle counts, resolve them
        # This should rarely happen, but prevents bugs
        if proton_count != self.protons or neutron_count != self.neutrons:
            logger.warning(f"Particle count mismatch. Expected {self.protons}p/{self.neutrons}n, got {proton_count}p/{neutron_count}n. Correcting...")
            
            # Handle excess protons by converting to neutrons
            if proton_count > self.protons:
                protons_to_convert = proton_count - self.protons
                for p in self.particles:
                    if p.type == ParticleType.PROTON and protons_to_convert > 0:
                        p.type = ParticleType.NEUTRON
                        protons_to_convert -= 1
                        if protons_to_convert == 0:
                            break
            
            # Handle excess neutrons by converting to protons
            if neutron_count > self.neutrons:
                neutrons_to_convert = neutron_count - self.neutrons
                for p in self.particles:
                    if p.type == ParticleType.NEUTRON and neutrons_to_convert > 0:
                        p.type = ParticleType.PROTON
                        neutrons_to_convert -= 1
                        if neutrons_to_convert == 0:
                            break
        
        # Recalculate stability based on new composition
        self.stability = self.calculate_stability()
        self.half_life = self.calculate_half_life()
        
        # Apply a very small damping force to all particles to gradually stabilize
        # This simulates the quantum mechanical ground state
        for p in self.particles:
            p.vx *= 0.8  # Stronger damping to stabilize velocities
            p.vy *= 0.8

    def update_center_of_mass(self):
        """Update the nucleus center position based on constituent particles"""
        if len(self.particles) > 0:
            self.x = sum(p.x for p in self.particles) / len(self.particles)
            self.y = sum(p.y for p in self.particles) / len(self.particles)

class KernelTimeoutError(Exception):
    """Exception raised when a kernel execution times out."""
    pass

class NuclearSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Nuclear Physics Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize OpenCL context and queue for GPU acceleration
        self.opencl_info = {}
        try:
            self.setup_opencl()
            self.gpu_available = True
            logger.info("GPU acceleration enabled")
        except Exception as e:
            logger.error(f"OpenCL initialization failed: {str(e)}")
            logger.info("Using CPU fallback.")
            self.gpu_available = False
            self.display_opencl_diagnostics()
        
        # Kernel execution monitoring
        self.kernel_timeout = 5.0  # seconds
        self.kernel_running = False
        
        # Simulation state
        self.nucleus = None
        self.particles = []
        self.time_scale = 1.0  # Simulation speed multiplier
        self.min_time_scale = 1e-40  # Close to Planck time scale
        self.max_time_scale = 1e10    # Very fast simulation
        self.time_scale_factor = 10.0  # Multiplier for time changes
        self.quantum_time_mode = False  # Flag for quantum time scale
        self.time_passed = 0
        self.decay_counts = {"ALPHA": 0, "BETA_MINUS": 0, "BETA_PLUS": 0, "GAMMA": 0}
        self.decay_times = deque(maxlen=100)  # Store last 100 decay times
        
        # Font for text display
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Create initial nucleus
        self.create_nucleus(92, 146)  # Uranium-238

        # Animation settings
        self.add_thermal_motion = False  # Disable thermal motion by default (more physically accurate)
        
        # Visualization settings - zoom and camera system
        self.zoom_level = 15.0  # Initial zoom level
        self.target_zoom = 15.0  # For smooth zoom transitions
        self.zoom_speed = 0.1  # Speed of zoom changes
        self.camera_x = SIMULATION_WIDTH / 2  # Camera position (world coordinates)
        self.camera_y = SIMULATION_HEIGHT / 2
        self.camera_target_x = SIMULATION_WIDTH / 2  # Target camera position
        self.camera_target_y = SIMULATION_HEIGHT / 2
        self.camera_smoothing = 0.1  # Camera smoothing factor
    
    def display_opencl_diagnostics(self):
        """Display detailed diagnostic information about OpenCL platforms and devices"""
        logger.info("\n--- OpenCL Diagnostics ---")
        try:
            platforms = cl.get_platforms()
            logger.info(f"Found {len(platforms)} OpenCL platform(s)")
            
            for i, platform in enumerate(platforms):
                logger.info(f"\nPlatform {i+1}: {platform.name}")
                logger.info(f"  Vendor: {platform.vendor}")
                logger.info(f"  Version: {platform.version}")
                
                try:
                    devices = platform.get_devices()
                    logger.info(f"  Found {len(devices)} device(s)")
                    
                    for j, device in enumerate(devices):
                        logger.info(f"  Device {j+1}: {device.name}")
                        logger.info(f"    Type: {cl.device_type.to_string(device.type)}")
                        logger.info(f"    Memory (global): {device.global_mem_size / (1024**2):.2f} MB")
                        logger.info(f"    Max compute units: {device.max_compute_units}")
                except Exception as device_error:
                    logger.error(f"  Error getting devices: {str(device_error)}")
            
            if not platforms:
                logger.warning("No OpenCL platforms found. Check your OpenCL installation.")
                
        except Exception as e:
            logger.error(f"Error during OpenCL diagnostics: {str(e)}")
    
    def setup_opencl(self):
        """Set up OpenCL context and compile kernels"""
        # Create OpenCL context
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        # Store information for diagnostics
        self.opencl_info["platforms"] = platforms
        
        # Try multiple approaches to create a context
        ctx = None
        device = None
        
        # First try: Use GPU device
        try:
            # Get all GPU devices from all platforms
            gpu_devices = []
            for platform in platforms:
                try:
                    platform_gpus = platform.get_devices(device_type=cl.device_type.GPU)
                    gpu_devices.extend(platform_gpus)
                except:
                    continue
            
            if gpu_devices:
                device = gpu_devices[0]  # Take first GPU device
                logger.info(f"Using GPU device: {device.name}")
                ctx = cl.Context([device])
        except Exception as e:
            logger.warning(f"Failed to create GPU context: {e}")
        
        # Second try: Use default context
        if ctx is None:
            try:
                ctx = cl.create_some_context(interactive=False)
                logger.info("Created default context")
            except Exception as e:
                logger.warning(f"Failed to create default context: {e}")
        
        # Third try: CPU fallback
        if ctx is None:
            try:
                cpu_devices = []
                for platform in platforms:
                    try:
                        platform_cpus = platform.get_devices(device_type=cl.device_type.CPU)
                        cpu_devices.extend(platform_cpus)
                    except:
                        continue
                
                if cpu_devices:
                    device = cpu_devices[0]  # Take first CPU device
                    logger.info(f"Using CPU device: {device.name}")
                    ctx = cl.Context([device])
            except Exception as e:
                logger.warning(f"Failed to create CPU context: {e}")
        
        # If all approaches fail, raise exception
        if ctx is None:
            raise RuntimeError("Failed to create OpenCL context using any available device")
        
        self.ctx = ctx
        
        # Create command queue with profiling enabled
        props = cl.command_queue_properties.PROFILING_ENABLE
        self.queue = cl.CommandQueue(self.ctx, properties=props)
        
        # Load and build OpenCL kernel with fixed nuclear forces
        kernel_src = """
        // Random number generator for OpenCL
        // Based on xorshift algorithm
        unsigned int rand_xorshift(unsigned int state)
        {
            // Xorshift algorithm from George Marsaglia's paper
            state ^= (state << 13);
            state ^= (state >> 17);
            state ^= (state << 5);
            return state;
        }
        
        // Convert unsigned int to float in [0,1) range
        float rand_float_func(unsigned int* state)
        {
            *state = rand_xorshift(*state);
            return (float)(*state) / (float)(0xffffffffU);  // 2^32-1
        }
        
        // Box-Muller transform for Gaussian distribution
        float rand_gaussian(unsigned int* state)
        {
            float u1 = rand_float_func(state);
            float u2 = rand_float_func(state);
            // Prevent log(0)
            if (u1 < 1e-6f) u1 = 1e-6f;
            
            // Box-Muller transform
            return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
        }
        
        // Update particles based on their velocities and forces
        __kernel void update_particles(__global float4* particles,
                                      __global float4* forces,
                                      int num_particles,
                                      float dt) {
            int gid = get_global_id(0);
            if (gid < num_particles) {
                // Update velocity based on applied forces
                particles[gid].z += forces[gid].x * dt;
                particles[gid].w += forces[gid].y * dt;
                
                // Apply stronger damping to avoid oscillations - critical for stability
                particles[gid].z *= 0.85f;  // Increased damping factor
                particles[gid].w *= 0.85f;
                
                // Then update position based on new velocity
                particles[gid].x += particles[gid].z * dt;
                particles[gid].y += particles[gid].w * dt;
            }
        }
        
        // Add thermal motion to particles using Gaussian noise for better quantum effects
        __kernel void add_thermal_motion(__global float4* particles,
                                       int num_particles,
                                       float strength,
                                       unsigned int seed) {
            int gid = get_global_id(0);
            if (gid < num_particles) {
                // Create a unique seed for each particle
                unsigned int state = seed + gid;
                
                // Generate Gaussian random values using Box-Muller transform
                float rx = rand_gaussian(&state) * strength;
                float ry = rand_gaussian(&state) * strength;
                
                // Add small random movement to position
                particles[gid].x += rx;
                particles[gid].y += ry;
            }
        }
        
        // Apply gentle force to prevent nucleus from flying apart completely but not forcing to center
        __kernel void apply_soft_containment(__global float4* particles,
                                          int num_particles,
                                          float2 nucleus_center,
                                          float region_radius,
                                          float strength,
                                          float dt) {
            int gid = get_global_id(0);
            if (gid >= num_particles) return;
            
            // Calculate distance from center
            float dx = particles[gid].x - nucleus_center.x;
            float dy = particles[gid].y - nucleus_center.y;
            float dist = sqrt(dx*dx + dy*dy);
            
            // ONLY apply containment when particles get very far from nucleus
            // This is a gentle force that prevents simulation objects from flying away
            if (dist > region_radius) {
                // Logarithmic force that increases with distance beyond threshold
                float force = strength * log(dist / region_radius);
                
                // Cap maximum force to prevent instability
                force = min(force, 5.0f);
                
                // Apply force toward center only if beyond threshold, and SCALE WITH DT
                if (dist > 0.0f) {
                    // Instead of directly modifying position, modify velocity
                    particles[gid].z -= dx * force / dist * (1.0f / dt);
                    particles[gid].w -= dy * force / dist * (1.0f / dt);
                }
            }
        }
        
        // Realistic nuclear potential implementation
        // Modified to include weak force for beta decay processes
        __kernel void physical_nuclear_forces(__global float4* particles,
                                           __global float4* forces,
                                           __global int* types,
                                           int num_particles,
                                           float2 nucleus_center,
                                           float strong_strength,
                                           float coulomb_strength,
                                           float pauli_strength,
                                           float gravity_strength,
                                           float weak_strength,
                                           float dt_factor) {
            int i = get_global_id(0);
            if (i >= num_particles) return;
            
            float totalFx = 0.0f;
            float totalFy = 0.0f;
            
            // Get particle position and type
            float px = particles[i].x;
            float py = particles[i].y;
            int type_i = types[i];  // 0 for proton, 1 for neutron
            
            // Limit interactions to avoid O(n²) performance issues
            int max_interactions = min(num_particles, 128);
            
            // Physical radius of nucleons - for hard-core repulsion
            float nucleon_radius = 2.5f;
            float max_force = 12.0f;  // Increased for stronger nuclear binding
            float epsilon = 0.15f;    // Better stability
            
            // Compute forces from all other particles
            for (int j = 0; j < max_interactions; j++) {
                if (i == j) continue;
                
                // Get other particle position and type
                float qx = particles[j].x;
                float qy = particles[j].y;
                int type_j = types[j];
                
                // Calculate vector between particles
                float dx = qx - px;
                float dy = qy - py;
                float dist2 = dx*dx + dy*dy;
                
                // Skip if too close (avoid division by zero)
                if (dist2 < 1e-10f) continue;
                
                float dist = sqrt(dist2);
                float net_force = 0.0f;
                
                // Hard-core repulsion - quantum mechanical effect preventing nucleons from overlapping
                float min_allowed_dist = nucleon_radius * 1.7f;
                if (dist < min_allowed_dist) {
                    float overlap = min_allowed_dist - dist;
                    float repulsion_strength = 60.0f;  // Increased for better stability
                    float repulsion = repulsion_strength * pow(overlap / min_allowed_dist, 1.5f);
                    
                    net_force -= repulsion;
                }
                
                // Nuclear force profile using Argonne v18 inspired potential
                float strong_range = 7.0f;  // Range increased for better binding (~3.5 fm)
                float r_ratio = dist / strong_range;
                
                float strong_force = 0.0f;
                
                if (dist < 2.8f) {
                    // Very short range - repulsive core
                    strong_force = -0.7f * strong_strength / (dist2 + epsilon);
                } else if (dist < 9.0f) {  // Extended range for better binding
                    // Mid range - attractive potential (major binding region)
                    // Enhanced mid-range attraction to maintain proper nuclear density
                    float yukawa = exp(-r_ratio) / (dist + epsilon);
                    float saturation = 0.9f;  // Increased saturation
                    strong_force = 1.25f * strong_strength * yukawa * saturation;  // 25% stronger attraction
                } else {
                    // Long range - exponential tail-off
                    strong_force = 0.15f * strong_strength * exp(-r_ratio * 1.8f) / (dist + epsilon);
                }
                
                // Cap strong force
                strong_force = clamp(strong_force, -max_force, max_force);
                
                net_force += strong_force;
                
                // ...rest of existing forces calculations...
                
                // Improved symmetry energy - enhanced to maintain proper n/p distribution
                if (type_i != type_j) {
                    float symmetry_range = 9.0f;  // Increased range
                    float symmetry_factor = exp(-dist / symmetry_range);
                    net_force += 0.2f * pauli_strength * symmetry_factor;  // Increased attraction between n-p
                }
                
                // ...existing code that applies the forces...
            }
            
            // Apply a gentler center-of-mass force that doesn't artificially compact the nucleus
            // This better represents collective nuclear motion
            float center_dx = nucleus_center.x - px;
            float center_dy = nucleus_center.y - py;
            float center_dist2 = center_dx*center_dx + center_dy*center_dy;
            float center_dist = sqrt(center_dist2);
            
            // Nuclear radius formula
            float nuclear_radius = 1.2f * pow(num_particles, 1.0f/3.0f) * 2.0f;
            
            if (center_dist > nuclear_radius * 1.5f) {
                // Only apply centering force when particles are too far from nucleus
                // This prevents artificial compaction while still maintaining cohesion
                float center_force = 0.03f * (center_dist - nuclear_radius);
                
                if (center_dist > 0.01f) {
                    // Add centering force, proportional to distance beyond expected radius
                    totalFx += center_force * center_dx / center_dist;
                    totalFy += center_force * center_dy / center_dist;
                }
            }
            
            // ...rest of existing code...
        }
        """
        try:
            self.program = cl.Program(self.ctx, kernel_src).build(options="-cl-fast-relaxed-math")
            logger.info("OpenCL program built successfully")
        except Exception as e:
            logger.error(f"Failed to build OpenCL program: {e}")
            try:
                self.program = cl.Program(self.ctx, kernel_src).build()
                logger.info("OpenCL program built successfully with fallback options")
            except Exception as e2:
                logger.error(f"Failed to build with fallback options: {e2}")
                raise
    
    def create_nucleus(self, protons, neutrons):
        """Create a new nucleus with the specified number of protons and neutrons"""
        self.nucleus = Nucleus(protons, neutrons, SIMULATION_WIDTH / 2, SIMULATION_HEIGHT / 2)
        self.particles = []  # Clear any existing decay particles
        self.time_passed = 0
        self.decay_counts = {"ALPHA": 0, "BETA_MINUS": 0, "BETA_PLUS": 0, "GAMMA": 0}
        self.decay_times = deque(maxlen=100)
        
        # Reset camera to focus on the new nucleus
        self.camera_target_x = self.nucleus.x
        self.camera_target_y = self.nucleus.y
    
    def update_camera(self, dt):
        """Smoothly update camera position and zoom level"""
        self.camera_x += (self.camera_target_x - self.camera_x) * self.camera_smoothing
        self.camera_y += (self.camera_target_y - self.camera_y) * self.camera_smoothing
        self.zoom_level += (self.target_zoom - self.zoom_level) * self.zoom_speed
    
    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates to screen coordinates"""
        screen_center_x = SIMULATION_WIDTH / 2
        screen_center_y = SIMULATION_HEIGHT / 2
        screen_x = screen_center_x + (world_x - self.camera_x) * self.zoom_level
        screen_y = screen_center_y + (world_y - self.camera_y) * self.zoom_level
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates"""
        screen_center_x = SIMULATION_WIDTH / 2
        screen_center_y = SIMULATION_HEIGHT / 2
        world_x = self.camera_x + (screen_x - screen_center_x) / self.zoom_level
        world_y = self.camera_y + (screen_y - screen_center_y) / self.zoom_level
        return world_x, world_y
    
    def format_time_scale(self):
        """Format the time scale in appropriate units"""
        if self.time_scale >= 1.0:
            return f"x{self.time_scale:.1f}"
        elif self.time_scale >= 1e-3:
            return f"x{self.time_scale:.3f}"
        elif self.time_scale >= 1e-15:  # Femtosecond scale
            return f"{self.time_scale/FEMTOSECOND:.1f} fs"
        elif self.time_scale >= 1e-30:  # Attosecond and below
            return f"{self.time_scale/ATTOSECOND:.1e} as"
        else:  # Approaching Planck time
            planck_ratio = self.time_scale / PLANCK_TIME
            return f"{planck_ratio:.1e} × Planck time"
    
    def update_simulation(self, dt):
        """Update the simulation state"""
        real_dt = dt * self.time_scale
        
        # Set quantum time mode flag for visual effects
        self.quantum_time_mode = self.time_scale < 1e-15
        
        # Special handling for quantum time scales
        if self.quantum_time_mode:
            # Use a minimum dt value to prevent instability
            real_dt = max(real_dt, 1e-10 / FPS)
            # In quantum mode, we might want to use a different physics model
            # This could be expanded later for quantum effects
        
        self.time_passed += real_dt
        self.update_camera(real_dt)
        
        # Update decays
        if self.nucleus and self.nucleus.should_decay(real_dt):
            new_particles = self.nucleus.decay()
            if new_particles:
                self.particles.extend(new_particles)
                self.nucleus.update_particles()
                last_decay = self.nucleus.decay_history[-1]
                if last_decay == DecayType.ALPHA:
                    self.decay_counts["ALPHA"] += 1
                elif last_decay == DecayType.BETA_MINUS:
                    self.decay_counts["BETA_MINUS"] += 1
                elif last_decay == DecayType.BETA_PLUS:
                    self.decay_counts["BETA_PLUS"] += 1
                elif last_decay == DecayType.GAMMA:
                    self.decay_counts["GAMMA"] += 1
                self.decay_times.append(self.time_passed)
        
        # Update decay particles - move them and handle lifetime
        particles_to_keep = []
        for particle in self.particles:
            # Update position based on velocity
            particle.x += particle.vx * real_dt
            particle.y += particle.vy * real_dt
            
            # Update age and check if particle should still exist
            particle.age += real_dt
            if particle.age < particle.lifetime:
                particles_to_keep.append(particle)
                
        # Remove expired particles
        self.particles = particles_to_keep
        
        # Update particle positions - GPU only
        if self.nucleus:
            self.update_nucleus_gpu(real_dt)
        
        # After GPU update, explicitly resolve any remaining overlaps
        if self.nucleus:
            self.resolve_overlapping_particles()
    
    def run_kernel_with_timeout(self, kernel_func, *args):
        """Run an OpenCL kernel with a timeout to prevent hangs"""
        result = [None]
        error = [None]
        self.kernel_running = True
        def kernel_thread():
            try:
                result[0] = kernel_func(*args)
            except Exception as e:
                error[0] = e
            finally:
                self.kernel_running = False
        thread = threading.Thread(target=kernel_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.kernel_timeout)
        if self.kernel_running:
            logger.warning(f"Kernel execution taking too long, may be frozen!")
            self.kernel_running = False
            raise KernelTimeoutError("Kernel execution timed out")
        if error[0]:
            raise error[0]
        return result[0]
    
    def update_nucleus_gpu(self, dt):
        """Update nucleus particle positions using GPU acceleration"""
        if not self.nucleus or len(self.nucleus.particles) == 0:
            return
        try:
            if len(self.nucleus.particles) > 1000:
                logger.warning("Very large nucleus detected, performance may suffer")
                
            num_particles = len(self.nucleus.particles)
            h_particles = np.zeros((num_particles, 4), dtype=np.float32)
            h_forces = np.zeros((num_particles, 4), dtype=np.float32)
            h_types = np.zeros(num_particles, dtype=np.int32)
            for i, p in enumerate(self.nucleus.particles):
                h_particles[i, 0] = p.x
                h_particles[i, 1] = p.y
                h_particles[i, 2] = p.vx  # Store current velocities
                h_particles[i, 3] = p.vy
                h_types[i] = 0 if p.type == ParticleType.PROTON else 1
            
            d_particles = cl.array.to_device(self.queue, h_particles)
            d_forces = cl.array.to_device(self.queue, h_forces)
            d_types = cl.array.to_device(self.queue, h_types)
            global_size = (num_particles,)
            local_size = None
            
            try:
                # Use time_scale as dt_factor for force scaling
                dt_factor = self.time_scale if self.time_scale > 1.0 else 1.0
                nucleus_center = np.array([self.nucleus.x, self.nucleus.y], dtype=np.float32)
                
                # Calculate forces with time scaling to prevent escaping at high speeds
                event = self.program.physical_nuclear_forces(
                    self.queue, global_size, local_size,
                    d_particles.data, d_forces.data, d_types.data,
                    np.int32(num_particles), nucleus_center,
                    np.float32(STRONG_FORCE_STRENGTH),
                    np.float32(COULOMB_STRENGTH),
                    np.float32(PAULI_STRENGTH),
                    np.float32(GRAVITY_STRENGTH),
                    np.float32(WEAK_FORCE_STRENGTH),
                    np.float32(dt_factor)  # Pass time scaling factor
                )
                event.wait()
                
                # Update positions based on calculated forces
                event = self.program.update_particles(
                    self.queue, global_size, local_size,
                    d_particles.data, d_forces.data,
                    np.int32(num_particles), np.float32(dt)
                )
                event.wait()
                
                # Soft containment to prevent nucleus from flying apart
                containment_radius = 5 * math.sqrt(self.nucleus.protons + self.nucleus.neutrons)
                containment_strength = 0.1
                event = self.program.apply_soft_containment(
                    self.queue, global_size, local_size,
                    d_particles.data, np.int32(num_particles),
                    nucleus_center, np.float32(containment_radius),
                    np.float32(containment_strength), np.float32(dt)
                )
                event.wait()
                
                # Add thermal motion if enabled
                if self.add_thermal_motion:
                    seed = np.uint32(int(time.time() * 1000) % 4294967295)
                    thermal_strength = 0.05 * dt
                    event = self.program.add_thermal_motion(
                        self.queue, global_size, local_size,
                        d_particles.data, np.int32(num_particles),
                        np.float32(thermal_strength), seed
                    )
                    event.wait()
                    
            except Exception as e:
                logger.error(f"Physics simulation failed: {str(e)}")
                return
                
            # Get results back from GPU
            h_particles_result = d_particles.get()
            for i, p in enumerate(self.nucleus.particles):
                p.x = h_particles_result[i, 0]
                p.y = h_particles_result[i, 1]
                p.vx = h_particles_result[i, 2]
                p.vy = h_particles_result[i, 3]
            
            # Apply additional velocity damping in Python for better stability
            # This helps reduce the "wibbling and wobbling"
            for p in self.nucleus.particles:
                # Apply stronger damping if velocity is high
                speed = math.sqrt(p.vx*p.vx + p.vy*p.vy)
                if speed > 15:  # High speed threshold
                    p.vx *= 0.7  # More aggressive damping for high speeds
                    p.vy *= 0.7
                
        except Exception as e:
            logger.error(f"GPU update failed: {str(e)}")

    def resolve_overlapping_particles(self):
        """Direct correction to prevent any particle overlap"""
        if not self.nucleus:
            return
            
        num_particles = len(self.nucleus.particles)
        min_allowed_dist = NUCLEON_RADIUS * 2.0
        
        # Check all pairs of particles
        for i in range(num_particles):
            p1 = self.nucleus.particles[i]
            for j in range(i+1, num_particles):
                p2 = self.nucleus.particles[j]
                
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dist2 = dx*dx + dy*dy
                
                # If overlap detected
                if dist2 < min_allowed_dist*min_allowed_dist:
                    dist = math.sqrt(dist2)
                    overlap = min_allowed_dist - dist
                    
                    # Direction vector
                    if dist > 0:
                        dx /= dist
                        dy /= dist
                    else:
                        # If particles are at the same position, move in random direction
                        angle = random.uniform(0, 2 * math.pi)
                        dx, dy = math.cos(angle), math.sin(angle)
                    
                    # Push particles apart slightly more than the overlap
                    push = overlap * 0.55  # Push each particle half the overlap + a bit extra
                    
                    # Move particles apart
                    p1.x -= dx * push 
                    p1.y -= dy * push
                    p2.x += dx * push
                    p2.y += dy * push
    
    def draw(self):
        """Draw the current simulation state with enhanced particle visuals"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw the femtometer ruler
        self.draw_ruler()
        
        # Create a list of particles sorted by Y position for proper rendering order
        if self.nucleus:
            # Sort particles by Y position to create a simple depth effect
            sorted_particles = sorted(self.nucleus.particles, key=lambda p: p.y)
            
            # Draw particles with visual distinction
            for particle in sorted_particles:
                screen_x, screen_y = self.world_to_screen(particle.x, particle.y)
                scaled_radius = max(1, int(particle.radius * self.zoom_level))
                
                if 0 <= screen_x < SIMULATION_WIDTH and 0 <= screen_y < SIMULATION_HEIGHT:
                    # Draw particle with visual distinction based on type
                    if particle.type == ParticleType.PROTON:
                        # Draw proton with highlight
                        pygame.draw.circle(self.screen, PROTON_COLOR, 
                                         (int(screen_x), int(screen_y)), 
                                         scaled_radius)
                        # Add small white highlight to give 3D appearance
                        highlight_radius = max(1, int(scaled_radius * 0.3))
                        highlight_offset = max(1, int(scaled_radius * 0.2))
                        pygame.draw.circle(self.screen, (255, 150, 150),
                                         (int(screen_x - highlight_offset), 
                                          int(screen_y - highlight_offset)),
                                         highlight_radius)
                    else:
                        # Draw neutron with a different pattern
                        pygame.draw.circle(self.screen, NEUTRON_COLOR, 
                                         (int(screen_x), int(screen_y)), 
                                         scaled_radius)
                        # Add a subtle ring to distinguish neutrons
                        if scaled_radius > 2:
                            pygame.draw.circle(self.screen, (150, 150, 200),
                                             (int(screen_x), int(screen_y)),
                                             scaled_radius - 1, 1)
        
        # Draw decay particles
        for particle in self.particles:
            screen_x, screen_y = self.world_to_screen(particle.x, particle.y)
            color = particle.get_color()
            if particle.lifetime < float('inf'):
                fade = 1.0 - min(particle.age / particle.lifetime, 1.0)
                color = tuple(int(c * fade) for c in color)
            scaled_radius = max(1, int(particle.radius * self.zoom_level))
            if (0 <= screen_x < SIMULATION_WIDTH and 0 <= screen_y < SIMULATION_HEIGHT):
                pygame.draw.circle(self.screen, color, 
                                 (int(screen_x), int(screen_y)), 
                                 scaled_radius)
        
        self.draw_info()
        pygame.display.flip()
    
    def draw_ruler(self):
        """Draw a ruler showing distances in femtometers"""
        # Calculate an appropriate ruler length based on zoom level
        # We want ruler to be about 20-25% of the screen width
        desired_screen_length = SIMULATION_WIDTH * 0.25
        
        # Calculate how many simulation units that is 
        sim_units_length = desired_screen_length / self.zoom_level
        
        # Convert to femtometers
        fm_length = sim_units_length * FM_PER_UNIT
        
        # Round to a nice number
        nice_lengths = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        nice_fm_length = nice_lengths[0]
        for length in nice_lengths:
            if length <= fm_length:
                nice_fm_length = length
        
        # Calculate the actual screen length of the ruler
        ruler_screen_length = (nice_fm_length / FM_PER_UNIT) * self.zoom_level
        
        # Position the ruler at the bottom of the screen
        ruler_x = 50
        ruler_y = SIMULATION_HEIGHT - 50
        
        # Draw the ruler line
        pygame.draw.line(self.screen, (200, 200, 200), 
                       (ruler_x, ruler_y), 
                       (ruler_x + ruler_screen_length, ruler_y), 
                       2)
        
        # Draw ticks at 20% intervals
        for i in range(6):
            tick_x = ruler_x + (ruler_screen_length * i / 5)
            tick_height = 10 if i % 5 == 0 else 5
            pygame.draw.circle(self.screen, (200, 200, 200), 
                             (int(tick_x), int(ruler_y)), 
                             2 if i % 5 == 0 else 1)
            pygame.draw.line(self.screen, (200, 200, 200),
                           (tick_x, ruler_y), 
                           (tick_x, ruler_y - tick_height),
                           1)
        
        # Label the ruler
        ruler_text = f"{nice_fm_length} fm"
        text_surf = self.font.render(ruler_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (ruler_x + ruler_screen_length/2 - text_surf.get_width()/2, ruler_y - 25))
        
        # Add a label explaining what it is
        label_text = "Scale: " 
        label_surf = self.font.render(label_text, True, (200, 200, 200))
        self.screen.blit(label_surf, (ruler_x - 10 - label_surf.get_width(), ruler_y - 10))
    
    def draw_info(self):
        """Draw information about the current simulation state"""
        info_x = SIMULATION_WIDTH + 20
        info_y = 20
        line_height = 25
        
        # Add femtometer scale info in the panel
        scale_text = f"1 simulation unit = {FM_PER_UNIT} fm"
        text = self.font.render(scale_text, True, (200, 200, 255))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        
        nucleon_text = f"Nucleon radius: {NUCLEON_RADIUS_FM} fm"
        text = self.font.render(nucleon_text, True, (200, 200, 255))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        
        # Continue with the rest of the existing info...
        accel_mode = "GPU" if self.gpu_available else "CPU"
        accel_color = (100, 255, 100) if self.gpu_available else (255, 100, 100)
        text = self.font.render(f"Acceleration: {accel_mode}", True, accel_color)
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        zoom_text = f"Zoom: {self.zoom_level:.1f}x"
        text = self.font.render(zoom_text, True, (200, 200, 255))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        if self.nucleus:
            element_name, symbol = self.get_element_name(self.nucleus.protons)
            text = self.font.render(f"Element: {element_name} ({symbol})", True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            mass_number = self.nucleus.protons + self.nucleus.neutrons
            text = self.font.render(f"Isotope: {symbol}-{mass_number}", True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Protons: {self.nucleus.protons}", True, (255, 100, 100))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Neutrons: {self.nucleus.neutrons}", True, (100, 100, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            stability_color = (
                int(255 * (1 - self.nucleus.stability)),
                int(255 * self.nucleus.stability),
                0
            )
            text = self.font.render(f"Stability: {self.nucleus.stability:.2f}", True, stability_color)
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            half_life_text = "Stable" if self.nucleus.half_life == float('inf') else f"{self.nucleus.half_life:.2e} years"
            text = self.font.render(f"Half-life: {half_life_text}", True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height * 2
            text = self.font.render("Decay Statistics:", True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Alpha: {self.decay_counts['ALPHA']}", True, ALPHA_COLOR)
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Beta-: {self.decay_counts['BETA_MINUS']}", True, ELECTRON_COLOR)
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Beta+: {self.decay_counts['BETA_PLUS']}", True, ELECTRON_COLOR)
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
            text = self.font.render(f"Gamma: {self.decay_counts['GAMMA']}", True, GAMMA_COLOR)
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height * 2
        
        # Modified time scale display
        time_scale_color = (255, 255, 255)
        if self.quantum_time_mode:
            time_scale_color = (100, 255, 255)  # Cyan for quantum time
            
        time_scale_text = f"Time Scale: {self.format_time_scale()}"
        text = self.font.render(time_scale_text, True, time_scale_color)
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        
        # Add quantum time indicator if in quantum mode
        if self.quantum_time_mode:
            quantum_text = "QUANTUM TIME SCALE"
            text = self.font.render(quantum_text, True, (100, 255, 255))
            self.screen.blit(text, (info_x, info_y))
            info_y += line_height
        
        text = self.font.render(f"Simulation Time: {self.time_passed:.1f}s", True, (255, 255, 255))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height * 2
        text = self.font.render("Controls:", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("Arrow Up/Down: Change time scale", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("Arrow Left/Right: Fine-tune time scale", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("0: Reset time scale", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("P: Set to Planck time scale", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("1-9: Select different isotopes", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("Space: Force decay", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("G: Toggle GPU acceleration", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("T: Toggle thermal motion", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("Q/E: Zoom in/out", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("WASD: Pan camera", True, (200, 200, 200))
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        info_y += line_height
        thermal_status = "ON" if self.add_thermal_motion else "OFF"
        thermal_color = (100, 255, 100) if self.add_thermal_motion else (255, 100, 100)
        text = self.font.render(f"Thermal Motion: {thermal_status}", True, thermal_color)
        self.screen.blit(text, (info_x, info_y))
        info_y += line_height
        text = self.font.render("Physics: Realistic", True, (255, 255, 100))
        self.screen.blit(text, (info_x, info_y))
    
    def get_element_name(self, atomic_number):
        """Get element name and symbol based on atomic number"""
        elements = {
            1: ("Hydrogen", "H"),
            2: ("Helium", "He"),
            6: ("Carbon", "C"),
            8: ("Oxygen", "O"),
            26: ("Iron", "Fe"),
            47: ("Silver", "Ag"),
            79: ("Gold", "Au"),
            82: ("Lead", "Pb"),
            92: ("Uranium", "U"),
            94: ("Plutonium", "Pu"),
        }
        if atomic_number in elements:
            return elements[atomic_number]
        return f"Element-{atomic_number}", f"E{atomic_number}"
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.nucleus:
                        new_particles = self.nucleus.decay()
                        if new_particles:
                            self.particles.extend(new_particles)
                            self.nucleus.update_particles()
                            last_decay = self.nucleus.decay_history[-1]
                            if last_decay == DecayType.ALPHA:
                                self.decay_counts["ALPHA"] += 1
                            elif last_decay == DecayType.BETA_MINUS:
                                self.decay_counts["BETA_MINUS"] += 1
                            elif last_decay == DecayType.BETA_PLUS:
                                self.decay_counts["BETA_PLUS"] += 1
                            elif last_decay == DecayType.GAMMA:
                                self.decay_counts["GAMMA"] += 1
                elif event.key == pygame.K_UP:
                    # Increase time scale by factor
                    self.time_scale *= self.time_scale_factor
                    self.time_scale = min(self.time_scale, self.max_time_scale)
                    logger.info(f"Time scale increased to {self.format_time_scale()}")
                elif event.key == pygame.K_DOWN:
                    # Decrease time scale by factor
                    self.time_scale /= self.time_scale_factor
                    self.time_scale = max(self.time_scale, self.min_time_scale)
                    logger.info(f"Time scale decreased to {self.format_time_scale()}")
                elif event.key == pygame.K_RIGHT:
                    # Fine-tune time scale increase
                    self.time_scale *= 2.0
                    self.time_scale = min(self.time_scale, self.max_time_scale)
                    logger.info(f"Time scale fine-tuned to {self.format_time_scale()}")
                elif event.key == pygame.K_LEFT:
                    # Fine-tune time scale decrease
                    self.time_scale /= 2.0
                    self.time_scale = max(self.time_scale, self.min_time_scale)
                    logger.info(f"Time scale fine-tuned to {self.format_time_scale()}")
                elif event.key == pygame.K_0:
                    # Reset time scale to normal
                    self.time_scale = 1.0
                    logger.info("Time scale reset to normal")
                elif event.key == pygame.K_p:
                    # Set to Planck time scale (ultra slow)
                    self.time_scale = self.min_time_scale
                    logger.info(f"Time scale set to near Planck time: {self.format_time_scale()}")
                elif event.key == pygame.K_1:
                    self.create_nucleus(1, 0)
                elif event.key == pygame.K_2:
                    self.create_nucleus(2, 2)
                elif event.key == pygame.K_3:
                    self.create_nucleus(6, 6)
                elif event.key == pygame.K_4:
                    self.create_nucleus(6, 8)
                elif event.key == pygame.K_5:
                    self.create_nucleus(26, 30)
                elif event.key == pygame.K_6:
                    self.create_nucleus(47, 60)
                elif event.key == pygame.K_7:
                    self.create_nucleus(79, 118)
                elif event.key == pygame.K_8:
                    self.create_nucleus(82, 126)
                elif event.key == pygame.K_9:
                    self.create_nucleus(92, 146)
                elif event.key == pygame.K_g:
                    if not self.gpu_available:
                        try:
                            self.setup_opencl()
                            self.gpu_available = True
                            logger.info("GPU acceleration enabled")
                        except Exception as e:
                            logger.error(f"Failed to re-enable GPU: {str(e)}")
                    else:
                        self.gpu_available = False
                        logger.info("GPU acceleration disabled")
                elif event.key == pygame.K_t:
                    self.add_thermal_motion = not self.add_thermal_motion
                    logger.info(f"Thermal motion {'enabled' if self.add_thermal_motion else 'disabled'}")
                elif event.key == pygame.K_q:
                    self.target_zoom *= 1.5
                    logger.info(f"Zooming in: {self.target_zoom:.1f}x")
                elif event.key == pygame.K_e:
                    self.target_zoom /= 1.5
                    self.target_zoom = max(0.5, self.target_zoom)
                    logger.info(f"Zooming out: {self.target_zoom:.1f}x")
                elif event.key == pygame.K_r:
                    self.target_zoom = 15.0
                    if self.nucleus:
                        self.camera_target_x = self.nucleus.x
                        self.camera_target_y = self.nucleus.y
        keys = pygame.key.get_pressed()
        move_speed = 5.0 / self.zoom_level
        if keys[pygame.K_w]:
            self.camera_target_y -= move_speed
        if keys[pygame.K_s]:
            self.camera_target_y += move_speed
        if keys[pygame.K_a]:
            self.camera_target_x -= move_speed
        if keys[pygame.K_d]:
            self.camera_target_x += move_speed
        
    def run(self):
        """Run the main simulation loop"""
        logger.info("Starting simulation")
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update_simulation(dt)
            self.draw()
        logger.info("Simulation ended")

if __name__ == "__main__":
    simulation = NuclearSimulation()
    simulation.run()
