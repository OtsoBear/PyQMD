import pygame
import numpy as np
import pyopencl as cl
import pyopencl.array
import random
import os
import math
import time
from collections import deque
import logging
import threading
from particles import ParticleType, Particle, Nucleus, DecayType
from decay_chains import get_decay_product, get_half_life
from nuclear_forces import NuclearForces
from rendering import Renderer

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NuclearSim")

try:
    import siphash24
except ImportError:
    logger.info("Installing siphash24...")
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "siphash24"])
    except Exception as e:
        logger.warning(f"Failed to install siphash24: {e}")

class NuclearSimulation:
    def __init__(self):
        pygame.init()
        # Make the window resizable
        self.screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
        pygame.display.set_caption("Nuclear Physics Simulation")
        self.clock = pygame.time.Clock()
        self.running = True
        
        try:
            self.forces = NuclearForces()
            self.gpu_available = True
        except Exception as e:
            logger.error(f"OpenCL initialization failed: {e}")
            self.gpu_available = False
        
        self.renderer = Renderer(self.screen)
        self.nucleus = None
        self.particles = []
        self.time_scale = 1.0
        # Dramatically increase max time scale capability
        self.min_time_scale = 1e-40
        self.max_time_scale = 1e30  # Allow ridiculous time scales (billions of years per second)
        self.time_passed = 0
        
        self.decay_counts = {decay_type.name: 0 for decay_type in DecayType if decay_type != DecayType.NONE}
        self.decay_times = deque(maxlen=100)
        
        self.physics_dt = 1.0 / 240.0
        self.fps_history = deque(maxlen=30)
        self.manual_accuracy = True
        self.accuracy = 0.8
        self.max_substeps = 20
        self.substeps_used = 0
        self.auto_adjust_substeps = True  # Auto-adjust substeps based on time scale
        self.physics_dt_factor = 0.8  # Used to scale physics timestep
        
        self.camera_pos = [400, 400]
        self.camera_target = [400, 400]
        # Fix zoom management
        self.zoom_level = 15.0
        self.target_zoom = 15.0
        self.zoom_speed = 0.1
        self.min_zoom = 0.1
        self.max_zoom = 100.0
        
        # Add more extreme time presets for large scale simulations
        self.time_scale_presets = {
            'real': 1.0,                  # Real-time
            'minute': 60.0,               # 1 minute per second
            'hour': 3600.0,               # 1 hour per second
            'day': 86400.0,               # 1 day per second
            'year': 31557600.0,           # 1 year per second
            'millennium': 31557600000.0,  # 1000 years per second
            'million': 31557600000000.0,  # 1 million years per second
            'billion': 31557600000000000.0, # 1 billion years per second
        }
        
        # Start with U-238, which is unstable
        self.create_nucleus(92, 146)
        
    def create_nucleus(self, protons, neutrons):
        self.nucleus = Nucleus(protons, neutrons, 400, 400)
        self.particles = []
        self.time_passed = 0
        self.decay_counts = {decay_type.name: 0 for decay_type in DecayType if decay_type != DecayType.NONE}
        self.decay_times = deque(maxlen=100)
        self.camera_target = [self.nucleus.x, self.nucleus.y]
        
        # Initialize decay chain with just the initial state, without creating a fake decay
        self.nucleus.decay_chain = []
        
        # Add initial isotope to decay chain (store just one element for initial state)
        element = self.get_element_symbol(protons)
        mass = protons + neutrons
        
        # Using a single tuple with just element and mass for initial state
        # We'll be checking for this special format to display differently
        # Add 0 for decay time since it's the initial state
        self.nucleus.decay_chain.append((element, mass, "-", element, mass, 0))
        
        # Set the last real decay time to current time (0 at start)
        self.nucleus.last_decay_time = self.time_passed
        
        # Ensure stability (half-life) is set properly
        self.nucleus.stability = get_half_life(protons, neutrons)
        
    def update_simulation(self, dt):
        current_fps = 1.0 / dt if dt > 0 else 60
        self.fps_history.append(current_fps)
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else current_fps
        
        desired_dt = dt * self.time_scale
        self.time_passed += desired_dt
        
        self.camera_pos[0] += (self.camera_target[0] - self.camera_pos[0]) * 0.1
        self.camera_pos[1] += (self.camera_target[1] - self.camera_pos[1]) * 0.1
        self.zoom_level += (self.target_zoom - self.zoom_level) * self.zoom_speed
        
        # Auto-adjust physics parameters based on time scale if enabled
        if self.auto_adjust_substeps and self.time_scale != 1.0:
            # Scale physics timestep inversely with time scale
            if self.time_scale > 1.0:
                # For faster simulation, increase timestep to reduce substeps
                physics_dt_scale = min(10.0, self.time_scale ** 0.3)
                adjusted_dt = self.physics_dt_factor * physics_dt_scale / 240.0
                self.physics_dt = min(1.0/60.0, adjusted_dt)  # Cap at 60 Hz
            else:
                # For slower simulation, decrease timestep for precision
                physics_dt_scale = max(0.1, self.time_scale ** 0.2)
                adjusted_dt = self.physics_dt_factor * physics_dt_scale / 240.0
                self.physics_dt = max(1.0/1000.0, adjusted_dt)  # Cap at 1000 Hz
        
        # Calculate desired physics timestep with accuracy factor
        effective_physics_dt = self.physics_dt * (2.0 - self.accuracy)
        
        # Calculate number of physics steps needed
        # If we're simulating very quickly, increase substeps to ensure accuracy
        time_scale_factor = 1.0 if self.time_scale <= 10.0 else math.log10(self.time_scale)
        adjusted_max_substeps = int(self.max_substeps * time_scale_factor) if self.auto_adjust_substeps else self.max_substeps
        
        # Calculate substeps needed
        num_steps = max(1, min(adjusted_max_substeps, int(desired_dt / effective_physics_dt)))
        self.substeps_used = num_steps
        
        # If we're hitting the max substeps limit consistently, log a warning
        if num_steps >= adjusted_max_substeps and adjusted_max_substeps > 0:
            if random.random() < 0.01:  # Only log occasionally to avoid spam
                logger.warning(f"Max substeps limit reached ({num_steps}/{adjusted_max_substeps}). Consider adjusting physics parameters.")
        
        for _ in range(num_steps):
            self.particles = [p for p in self.particles if self.update_particle(p, effective_physics_dt, desired_dt/num_steps)]
            
            # Fixed decay probability calculation
            step_time = desired_dt / num_steps
            if self.nucleus and self.nucleus.should_decay(step_time):
                self.handle_decay()
            
            if self.nucleus and len(self.nucleus.particles) > 0:
                if self.gpu_available:
                    self.forces.update_particles_gpu(self.nucleus.particles, effective_physics_dt)
                else:
                    self.forces.update_particles_cpu(self.nucleus.particles, effective_physics_dt)
                
        if self.nucleus:
            self.resolve_overlaps()
    
    def update_particle(self, particle, dt, age_dt):
        """Update particle positions and ages with complete time-scale independence"""
        
        # For decay particles, use a completely fixed animation speed
        # This ensures they always move at the same visual speed regardless of time scale
        if particle.type in [ParticleType.ALPHA, ParticleType.ELECTRON, 
                           ParticleType.GAMMA, ParticleType.POSITRON]:
            # Use a completely fixed dt for visual consistency
            ANIMATION_DT = 1.0/60.0  # Fixed 60fps-equivalent for smooth animation
            
            # Update position with fixed speed
            particle.x += particle.vx * ANIMATION_DT
            particle.y += particle.vy * ANIMATION_DT
            
            # Age still uses the simulation time step to ensure proper lifetime
            particle.age += age_dt
            
            # Check if particle should be removed
            return particle.age < particle.lifetime
        else:
            # For nucleus particles, use time-scaled dt
            effective_dt = dt * (self.time_scale ** 0.5)
            particle.x += particle.vx * effective_dt
            particle.y += particle.vy * effective_dt
            particle.age += age_dt
            return True  # Nucleus particles don't expire
    
    def handle_decay(self):
        p, n, decay_type, products = get_decay_product(self.nucleus.protons, self.nucleus.neutrons)
        
        if decay_type:
            # Get element symbols for decay chain tracking
            old_z, old_n = self.nucleus.protons, self.nucleus.neutrons
            old_element = self.get_element_symbol(old_z)
            new_element = self.get_element_symbol(p)
            old_mass = old_z + old_n
            new_mass = p + n
            
            # Initialize decay_chain if it doesn't exist
            if not hasattr(self.nucleus, 'decay_chain'):
                self.nucleus.decay_chain = []
                self.nucleus.last_decay_time = self.time_passed
            
            # Get a realistic decay time based on the isotope's half-life
            # instead of just using the simulation time difference
            current_time = self.time_passed
            last_time = getattr(self.nucleus, 'last_decay_time', current_time)
            
            # Calculate a realistic decay time based on half-life
            # Use the actual half-life of the decaying isotope
            half_life = self.nucleus.stability
            
            # If zero time has passed or half-life is extremely small,
            # generate a realistic time based on the half-life
            measured_time = current_time - last_time
            
            # For zero or near-zero measured times, use a statistical approach
            if measured_time < 0.001 or half_life < 0.001:
                # Use a random fraction of the half-life for a more realistic distribution
                # This follows the exponential decay probability distribution
                if half_life == float('inf'):
                    # For stable isotopes (shouldn't happen in normal decays)
                    decay_duration = 0
                else:
                    # For unstable isotopes, use a realistic statistical model
                    # based on the exponential decay formula
                    random_factor = -math.log(random.random())  # Exponential distribution
                    decay_duration = min(half_life * random_factor / 0.693, measured_time or half_life)
            else:
                # Use the actual measured time if it's significant
                decay_duration = measured_time
                
            # Record the actual decay step with the realistic time
            decay_type_symbol = self.get_decay_symbol(decay_type)
            
            # Ensure decay type symbol is a valid Unicode character
            if decay_type == DecayType.ALPHA:
                decay_type_symbol = "α"
            elif decay_type == DecayType.BETA_MINUS:
                decay_type_symbol = "β-"
            elif decay_type == DecayType.BETA_PLUS:
                decay_type_symbol = "β+"
            elif decay_type == DecayType.GAMMA:
                decay_type_symbol = "γ"
            
            # Add decay to chain with the realistic decay time
            self.nucleus.decay_chain.append((
                str(old_element), 
                int(old_mass), 
                str(decay_type_symbol), 
                str(new_element), 
                int(new_mass),
                decay_duration  # Use the realistic decay time
            ))
            
            # Update last decay time
            self.nucleus.last_decay_time = current_time
            
            # Format decay time for logging with appropriate units
            time_str = self.format_time_value_with_unit(decay_duration)
            logger.info(f"DECAY: {old_element}-{old_mass} → {new_element}-{new_mass} ({decay_type_symbol}) after {time_str}")
            
            # Update nucleus properties with new values
            self.nucleus.protons = p
            self.nucleus.neutrons = n
            self.nucleus.adjust_particles(decay_type)
            self.nucleus.update_center_of_mass()
            
            # Create decay products with consistent visualization
            decay_products = products(self.nucleus.x, self.nucleus.y)
            for product in decay_products:
                # Define fixed base speeds for different particle types for visual consistency
                if product.type == ParticleType.ALPHA:
                    base_speed = 80.0  # Alpha particles are relatively slow
                elif product.type == ParticleType.GAMMA:
                    base_speed = 180.0  # Gamma rays are very fast
                elif product.type in [ParticleType.ELECTRON, ParticleType.POSITRON]:
                    base_speed = 150.0  # Beta particles are quite fast
                else:
                    base_speed = 100.0  # Default for other particles
                    
                # Normalize the velocity vector but keep the direction
                velocity_mag = math.sqrt(product.vx**2 + product.vy**2)
                if velocity_mag > 0.001:
                    # Normalize and scale by the base speed for the particle type
                    product.vx = (product.vx / velocity_mag) * base_speed
                    product.vy = (product.vy / velocity_mag) * base_speed
                
                # Set a minimum lifetime for visibility regardless of time scale
                product.lifetime = max(0.5, product.lifetime)
            
            self.particles.extend(decay_products)
            self.decay_times.append(self.time_passed)
            
            # Set the new nucleus stability
            self.nucleus.stability = get_half_life(self.nucleus.protons, self.nucleus.neutrons)
    
    def resolve_overlaps(self):
        particles = self.nucleus.particles
        min_dist = 5.0  # 2 * radius
        
        for i in range(len(particles)):
            for j in range(i+1, len(particles)):
                dx = particles[j].x - particles[i].x
                dy = particles[j].y - particles[i].y
                dist2 = dx*dx + dy*dy
                
                if dist2 < min_dist*min_dist:
                    dist = math.sqrt(dist2)
                    if dist < 0.001:
                        angle = random.uniform(0, 2 * math.pi)
                        dx, dy = math.cos(angle), math.sin(angle)
                        dist = 0.001
                    else:
                        dx /= dist
                        dy /= dist
                        
                    push = (min_dist - dist) * 0.5
                    particles[i].x -= dx * push
                    particles[i].y -= dy * push
                    particles[j].x += dx * push
                    particles[j].y += dy * push
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize event
                self.handle_resize(event.size)
            elif event.type == pygame.MOUSEWHEEL:
                # Check if mouse is over info panel for scrolling
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x > self.renderer.simulation_width:
                    # Mouse is over info panel
                    
                    # Calculate position for decay chain section (approximate)
                    # This is a rough estimate; you may need to adjust based on your layout
                    info_panel_height = self.renderer.height
                    decay_chain_start = 250  # Approximate Y position where decay chain starts
                    
                    if mouse_y > decay_chain_start and hasattr(self.nucleus, 'decay_chain') and len(self.nucleus.decay_chain) > 1:
                        # Mouse is over decay chain section
                        self.renderer.handle_scroll(-event.y * 1, section="decay_chain")
                    else:
                        # Mouse is over other parts of info panel
                        self.renderer.handle_scroll(-event.y * 30)
                else:
                    # Mouse over simulation - handle zooming
                    if event.y > 0:
                        self.target_zoom *= 1.2
                    elif event.y < 0:
                        self.target_zoom /= 1.2
                    self.target_zoom = max(self.min_zoom, min(self.max_zoom, self.target_zoom))
        
        keys = pygame.key.get_pressed()
        move_speed = 5.0 / self.zoom_level
        if keys[pygame.K_w]: self.camera_target[1] -= move_speed
        if keys[pygame.K_s]: self.camera_target[1] += move_speed
        if keys[pygame.K_a]: self.camera_target[0] -= move_speed
        if keys[pygame.K_d]: self.camera_target[0] += move_speed
    
    def handle_resize(self, size):
        """Handle window resize event"""
        width, height = size
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        # Inform renderer of new window size
        self.renderer.resize(width, height)
    
    def handle_keypress(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE and self.nucleus:
            self.handle_decay()
            
        # Time scale controls with more options
        elif key == pygame.K_UP:
            self.time_scale = min(self.time_scale * 10.0, self.max_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_DOWN:
            self.time_scale = max(self.time_scale / 10.0, self.min_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_RIGHT:
            self.time_scale = min(self.time_scale * 2.0, self.max_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_LEFT:
            self.time_scale = max(self.time_scale / 2.0, self.min_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_0:
            self.time_scale = 1.0  # Real-time
            logger.info("Time scale: real-time")
            
        # Enhance time scale presets
        elif key == pygame.K_r:  # Real-time
            self.time_scale = self.time_scale_presets['real']
            logger.info("Time scale: real-time")
        elif key == pygame.K_t:  # Time compression - minute per second
            self.time_scale = self.time_scale_presets['minute']
            logger.info("Time scale: 1 minute per second")
        elif key == pygame.K_h:  # Hour per second
            self.time_scale = self.time_scale_presets['hour']
            logger.info("Time scale: 1 hour per second")
        elif key == pygame.K_j:  # Day per second (don't bind to D which is used for camera)
            self.time_scale = self.time_scale_presets['day']
            logger.info("Time scale: 1 day per second")
        elif key == pygame.K_y:  # Year per second
            self.time_scale = self.time_scale_presets['year']
            logger.info("Time scale: 1 year per second")
        elif key == pygame.K_m:  # Millennium per second
            self.time_scale = self.time_scale_presets['millennium']
            logger.info("Time scale: 1000 years per second")
        elif key == pygame.K_b:  # Billion years per second
            self.time_scale = self.time_scale_presets['billion']
            logger.info("Time scale: 1 billion years per second")
            
        # Fix zoom controls
        elif key == pygame.K_q:
            self.target_zoom = min(self.max_zoom, self.target_zoom * 1.5)
            logger.info(f"Zooming in: {self.target_zoom:.1f}x")
        elif key == pygame.K_e:
            self.target_zoom = max(self.min_zoom, self.target_zoom / 1.5)
            logger.info(f"Zooming out: {self.target_zoom:.1f}x")
        elif key == pygame.K_z:  # Reset zoom
            self.target_zoom = 15.0
            logger.info("Zoom reset to default")
            
        elif key == pygame.K_f:  # Changed from A to F
            # Toggle automatic substep adjustment
            self.auto_adjust_substeps = not self.auto_adjust_substeps
            logger.info(f"Auto-adjust substeps: {'ON' if self.auto_adjust_substeps else 'OFF'}")
            
        elif key >= pygame.K_1 and key <= pygame.K_9:
            # Replace with unstable isotopes
            isotopes = {
                pygame.K_1: (1, 2),     # H-3 (tritium) - unstable
                pygame.K_2: (2, 3),     # He-5 - unstable with short half-life
                pygame.K_3: (6, 8),     # C-14 - unstable, carbon dating
                pygame.K_4: (8, 9),     # O-17 - slightly unstable
                pygame.K_5: (26, 33),   # Fe-59 - unstable
                pygame.K_6: (47, 61),   # Ag-108 - unstable
                pygame.K_7: (79, 119),  # Au-198 - unstable
                pygame.K_8: (82, 127),  # Pb-209 - unstable
                pygame.K_9: (92, 146),  # U-238 - unstable
            }
            if key in isotopes:
                self.create_nucleus(*isotopes[key])
                
        # Add key to reset decay chain scroll
        elif key == pygame.K_c:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.decay_chain_scroll = 0
                logger.info("Decay chain scroll reset to top")
                
        # Add keys for scrolling the decay chain
        elif key == pygame.K_PAGEUP:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.handle_scroll(-5, section="decay_chain")
                logger.info("Scrolling decay chain up")
        elif key == pygame.K_PAGEDOWN:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.handle_scroll(5, section="decay_chain")
                logger.info("Scrolling decay chain down")

    def get_element_symbol(self, atomic_number):
        """Get the element symbol for an atomic number"""
        elements = {
            1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
            9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
            16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
            23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
            30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
            37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
            44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
            51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
            58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
            65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
            72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
            79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
            86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
            93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es",
            100: "Fm", 101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db",
            106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg",
            112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts",
            118: "Og"
        }
        return elements.get(atomic_number, f"E{atomic_number}")
    
    def get_decay_symbol(self, decay_type):
        """Get a symbol for the decay type"""
        symbols = {
            DecayType.ALPHA: "α",
            DecayType.BETA_MINUS: "β-",
            DecayType.BETA_PLUS: "β+",
            DecayType.GAMMA: "γ",
            DecayType.NEUTRON_EMISSION: "n",
            DecayType.PROTON_EMISSION: "p",
            DecayType.SPONTANEOUS_FISSION: "SF"
        }
        return symbols.get(decay_type, "?")
    
    def format_time_value_with_unit(self, seconds):
        """Format a time value with appropriate units based on scale"""
        abs_seconds = abs(seconds)
        if abs_seconds == 0:
            return "0 s"
        elif abs_seconds < 1e-15:
            return f"{seconds * 1e18:.2f} as"  # attoseconds
        elif abs_seconds < 1e-12:
            return f"{seconds * 1e15:.2f} fs"  # femtoseconds
        elif abs_seconds < 1e-9:
            return f"{seconds * 1e12:.2f} ps"  # picoseconds
        elif abs_seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"   # nanoseconds
        elif abs_seconds < 1e-3:
            return f"{seconds * 1e6:.2f} μs"   # microseconds
        elif abs_seconds < 1:
            return f"{seconds * 1e3:.2f} ms"   # milliseconds
        elif abs_seconds < 60:
            return f"{seconds:.2f} s"          # seconds
        elif abs_seconds < 3600:
            return f"{seconds / 60:.2f} min"   # minutes
        elif abs_seconds < 86400:
            return f"{seconds / 3600:.2f} h"   # hours
        elif abs_seconds < 31557600:
            return f"{seconds / 86400:.2f} days"  # days
        else:
            return f"{seconds / 31557600:.2f} years"  # years
    
    def run(self):
        logger.info("Starting simulation")
        last_time = time.time()
        try:
            while self.running:
                dt = min(self.clock.tick(60) / 1000.0, 0.1)
                
                self.handle_events()
                self.update_simulation(dt)
                self.renderer.render(self.nucleus, self.particles, 
                                    self.camera_pos, self.zoom_level, 
                                    self.time_scale, self.accuracy, 
                                    self.physics_dt, self.substeps_used,
                                    self.max_substeps, self.gpu_available,
                                    self.decay_counts, self.time_passed)
                
                time.sleep(max(0, (1.0/60.0) - (time.time() - last_time)))
                last_time = time.time()
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            pygame.quit()
            logger.info("Simulation ended")

if __name__ == "__main__":
    simulation = NuclearSimulation()
    simulation.run()


