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
        self.min_time_scale = 1e-40
        self.max_time_scale = 1e30
        self.time_passed = 0
        
        self.decay_counts = {decay_type.name: 0 for decay_type in DecayType if decay_type != DecayType.NONE}
        self.decay_times = deque(maxlen=100)
        
        # Increased default dt for better performance
        self.physics_dt = 1.0 / 120.0
        self.fps_history = deque(maxlen=30)
        self.manual_accuracy = True
        self.accuracy = 1
        self.max_substeps = 20
        self.substeps_used = 0
        self.auto_adjust_substeps = False
        self.physics_dt_factor = 0.8
        
        self.camera_pos = [400, 400]
        self.camera_target = [400, 400]
        self.zoom_level = 15.0
        self.target_zoom = 15.0
        self.zoom_speed = 0.12  # Slightly faster for smoother experience
        self.min_zoom = 0.1
        self.max_zoom = 100.0
        
        # Input mode for custom nucleus creation
        self.input_mode = False
        self.input_values = [92, 146]  # Default U-238
        self.input_cursor = 0  # 0 for protons, 1 for neutrons
        
        # Start with U-238, which is unstable
        self.create_nucleus(92, 146)
        
    def create_nucleus(self, protons, neutrons):
        self.nucleus = Nucleus(protons, neutrons, 400, 400)
        self.particles = []
        self.time_passed = 0
        self.decay_counts = {decay_type.name: 0 for decay_type in DecayType if decay_type != DecayType.NONE}
        self.decay_times = deque(maxlen=100)
        self.camera_target = [self.nucleus.x, self.nucleus.y]
        
        self.nucleus.decay_chain = []
        element = self.get_element_symbol(protons)
        mass = protons + neutrons
        self.nucleus.decay_chain.append((element, mass, "-", element, mass, 0))
        self.nucleus.last_decay_time = self.time_passed
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
        
        if self.auto_adjust_substeps and self.time_scale != 1.0:
            if self.time_scale > 1.0:
                physics_dt_scale = min(10.0, self.time_scale ** 0.3)
                adjusted_dt = self.physics_dt_factor * physics_dt_scale / 240.0
                self.physics_dt = min(1.0/60.0, adjusted_dt)
            else:
                physics_dt_scale = max(0.1, self.time_scale ** 0.2)
                adjusted_dt = self.physics_dt_factor * physics_dt_scale / 240.0
                self.physics_dt = max(1.0/1000.0, adjusted_dt)
        
        effective_physics_dt = self.physics_dt * (2.0 - self.accuracy)
        
        time_scale_factor = 1.0 if self.time_scale <= 10.0 else math.log10(self.time_scale)
        adjusted_max_substeps = int(self.max_substeps * time_scale_factor) if self.auto_adjust_substeps else self.max_substeps
        
        num_steps = max(1, min(adjusted_max_substeps, int(desired_dt / effective_physics_dt)))
        self.substeps_used = num_steps
        
        if num_steps >= adjusted_max_substeps and adjusted_max_substeps > 0:
            if random.random() < 0.01:
                logger.warning(f"Max substeps limit reached ({num_steps}/{adjusted_max_substeps}). Consider adjusting physics parameters.")
        
        for _ in range(num_steps):
            self.particles = [p for p in self.particles if self.update_particle(p, effective_physics_dt, desired_dt/num_steps)]
            
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
        if particle.type in [ParticleType.ALPHA, ParticleType.ELECTRON, 
                           ParticleType.GAMMA, ParticleType.POSITRON]:
            ANIMATION_DT = 1.0/240.0
            
            substep_factor = 10.0 / max(1.0, self.substeps_used)
            SPEED_SCALE = 0.3 * substep_factor
            
            particle.x += particle.vx * ANIMATION_DT * SPEED_SCALE
            particle.y += particle.vy * ANIMATION_DT * SPEED_SCALE
            
            aging_scale = min(1.0, 1.0 / (math.sqrt(max(1.0, self.time_scale / 100.0)) * 
                                         math.sqrt(max(1.0, self.substeps_used / 10.0))))
            particle.age += age_dt * aging_scale
            
            return particle.age < particle.lifetime
        else:
            effective_dt = dt * (self.time_scale ** 0.5)
            particle.x += particle.vx * effective_dt
            particle.y += particle.vy * effective_dt
            particle.age += age_dt
            return True
    
    def handle_decay(self):
        p, n, decay_type, products = get_decay_product(self.nucleus.protons, self.nucleus.neutrons)
        
        if decay_type:
            old_z, old_n = self.nucleus.protons, self.nucleus.neutrons
            old_element = self.get_element_symbol(old_z)
            new_element = self.get_element_symbol(p)
            old_mass = old_z + old_n
            new_mass = p + n
            
            if not hasattr(self.nucleus, 'decay_chain'):
                self.nucleus.decay_chain = []
                self.nucleus.last_decay_time = self.time_passed
            
            current_time = self.time_passed
            last_time = getattr(self.nucleus, 'last_decay_time', current_time)
            
            half_life = self.nucleus.stability
            measured_time = current_time - last_time
            
            if measured_time < 0.001 or half_life < 0.001:
                if half_life == float('inf'):
                    decay_duration = 0
                else:
                    random_factor = -math.log(random.random())
                    decay_duration = min(half_life * random_factor / 0.693, measured_time or half_life)
            else:
                decay_duration = measured_time
                
            decay_type_symbol = self.get_decay_symbol(decay_type)
            
            if decay_type == DecayType.ALPHA:
                decay_type_symbol = "α"
            elif decay_type == DecayType.BETA_MINUS:
                decay_type_symbol = "β-"
            elif decay_type == DecayType.BETA_PLUS:
                decay_type_symbol = "β+"
            elif decay_type == DecayType.GAMMA:
                decay_type_symbol = "γ"
            
            self.nucleus.decay_chain.append((
                str(old_element), 
                int(old_mass), 
                str(decay_type_symbol), 
                str(new_element), 
                int(new_mass),
                decay_duration
            ))
            
            self.nucleus.last_decay_time = current_time
            
            time_str = self.format_time_value_with_unit(decay_duration)
            logger.info(f"DECAY: {old_element}-{old_mass} → {new_element}-{new_mass} ({decay_type_symbol}) after {time_str}")
            
            self.nucleus.protons = p
            self.nucleus.neutrons = n
            self.nucleus.adjust_particles(decay_type)
            self.nucleus.update_center_of_mass()
            
            decay_products = products(self.nucleus.x, self.nucleus.y)
            for product in decay_products:
                if product.type == ParticleType.ALPHA:
                    base_speed = 30.0
                elif product.type == ParticleType.GAMMA:
                    base_speed = 60.0
                elif product.type in [ParticleType.ELECTRON, ParticleType.POSITRON]:
                    base_speed = 50.0
                else:
                    base_speed = 40.0
                    
                velocity_mag = math.sqrt(product.vx**2 + product.vy**2)
                if velocity_mag > 0.001:
                    substep_multiplier = max(1.0, self.substeps_used / 10.0)
                    product.vx = (product.vx / velocity_mag) * base_speed
                    product.vy = (product.vy / velocity_mag) * base_speed
                
                base_lifetime = 5.0
                
                if self.time_scale > 1.0:
                    time_scale_factor = max(1.0, self.time_scale / 100.0)
                    substep_factor = max(1.0, math.sqrt(self.substeps_used))
                    dt_factor = max(1.0, 0.016 / self.physics_dt)
                    combined_factor = time_scale_factor * substep_factor * dt_factor
                    min_lifetime = base_lifetime * substep_factor
                    max_lifetime = 12000.0
                    product.lifetime = max(min_lifetime, base_lifetime * combined_factor)
                    
                    if self.substeps_used > 15:
                        product.lifetime *= (self.substeps_used / 15.0)
                else:
                    product.lifetime = max(product.lifetime, base_lifetime * max(1.0, self.substeps_used / 5.0))
                
                if random.random() < 0.05:
                    substep_info = f", substeps: {self.substeps_used}"
                    logger.info(f"Particle {product.type.name} lifetime: {product.lifetime:.2f}s{substep_info}")
            
            self.particles.extend(decay_products)
            self.decay_times.append(self.time_passed)
            
            self.nucleus.stability = get_half_life(self.nucleus.protons, self.nucleus.neutrons)
    
    def resolve_overlaps(self):
        particles = self.nucleus.particles
        min_dist = 5.0
        
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
                if self.input_mode:
                    self.handle_input_keypress(event)
                else:
                    self.handle_keypress(event.key)
            elif event.type == pygame.VIDEORESIZE:
                self.handle_resize(event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle section toggle clicks
                if event.button == 1:  # Left mouse button
                    if hasattr(self.renderer, 'handle_mouse_click'):
                        self.renderer.handle_mouse_click()
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                decay_chain_x = self.renderer.width - 320
                
                if mouse_x >= decay_chain_x:
                    self.renderer.handle_scroll(-event.y * 3, section="decay_chain")
                elif mouse_x > self.renderer.simulation_width:
                    self.renderer.handle_scroll(-event.y * 30)
                else:
                    if event.y > 0:
                        self.target_zoom *= 1.2
                    elif event.y < 0:
                        self.target_zoom /= 1.2
                    self.target_zoom = max(self.min_zoom, min(self.max_zoom, self.target_zoom))
        
        if not self.input_mode:
            keys = pygame.key.get_pressed()
            move_speed = 5.0 / self.zoom_level
            if keys[pygame.K_w]: self.camera_target[1] -= move_speed
            if keys[pygame.K_s]: self.camera_target[1] += move_speed
            if keys[pygame.K_a]: self.camera_target[0] -= move_speed
            if keys[pygame.K_d]: self.camera_target[0] += move_speed
    
    def handle_resize(self, size):
        width, height = size
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.renderer.resize(width, height)
        
        if hasattr(self.renderer, 'decay_chain_scroll'):
            self.renderer.decay_chain_scroll = 0
    
    def handle_input_keypress(self, event):
        if event.key == pygame.K_ESCAPE:
            self.input_mode = False
            return
        
        if event.key == pygame.K_RETURN:
            try:
                p, n = self.input_values
                if 0 < p <= 118 and n > 0:
                    self.create_nucleus(p, n)
                    self.input_mode = False
                else:
                    logger.warning("Invalid nucleus parameters")
            except Exception as e:
                logger.error(f"Error creating nucleus: {e}")
            return
            
        if event.key == pygame.K_TAB:
            self.input_cursor = (self.input_cursor + 1) % 2
            return
            
        if event.key == pygame.K_BACKSPACE:
            value = self.input_values[self.input_cursor]
            self.input_values[self.input_cursor] = value // 10
            return
            
        if event.unicode.isdigit():
            digit = int(event.unicode)
            current = self.input_values[self.input_cursor]
            new_val = current * 10 + digit
            if (self.input_cursor == 0 and new_val <= 118) or new_val <= 200:
                self.input_values[self.input_cursor] = new_val

    def handle_keypress(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE and self.nucleus:
            self.handle_decay()
            
        # Time scale controls
        elif key == pygame.K_UP:
            self.time_scale = min(self.time_scale * 10.0, self.max_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_DOWN:
            self.time_scale = max(self.time_scale / 10.0, self.min_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_i:
            self.time_scale = min(self.time_scale * 2.0, self.max_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_k:
            self.time_scale = max(self.time_scale / 2.0, self.min_time_scale)
            logger.info(f"Time scale: {self.time_scale}")
        elif key == pygame.K_o:
            self.time_scale = 1.0
            logger.info("Time scale: real-time")
            
        # Physics simulation controls
        elif key == pygame.K_p:
            self.max_substeps = min(100, self.max_substeps + 5)
            logger.info(f"Max substeps: {self.max_substeps}")
        elif key == pygame.K_l:
            self.max_substeps = max(1, self.max_substeps - 5)
            logger.info(f"Max substeps: {self.max_substeps}")
        elif key == pygame.K_x:
            self.physics_dt *= 1.1
            logger.info(f"Physics dt: {self.physics_dt}")
        elif key == pygame.K_z:
            self.physics_dt /= 1.1
            logger.info(f"Physics dt: {self.physics_dt}")
            
        # Zoom controls
        elif key == pygame.K_q:
            self.target_zoom = min(self.max_zoom, self.target_zoom * 1.5)
        elif key == pygame.K_e:
            self.target_zoom = max(self.min_zoom, self.target_zoom / 1.5)
        elif key == pygame.K_r:
            self.target_zoom = 15.0
            
        # Adjust grid scale
        elif key == pygame.K_g:
            # Increase grid size (in femtometers)
            self.renderer.fm_per_grid *= 2.0
            if self.renderer.fm_per_grid > 32.0:
                self.renderer.fm_per_grid = 1.0
            logger.info(f"Grid scale: {self.renderer.fm_per_grid} femtometers")
            
        elif key == pygame.K_f:
            self.auto_adjust_substeps = not self.auto_adjust_substeps
            logger.info(f"Auto-adjust substeps: {'ON' if self.auto_adjust_substeps else 'OFF'}")
            
        # Tab navigation
        elif key == pygame.K_TAB:
            # Cycle through info panel tabs
            self.renderer.active_section = (self.renderer.active_section + 1) % self.renderer.total_sections
            
        # Isotope selection
        elif key >= pygame.K_1 and key <= pygame.K_9:
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
                
        # Custom nucleus input
        elif key == pygame.K_n:
            self.input_mode = True
            self.input_values = [0, 0]
            self.input_cursor = 0
                
        # Decay chain navigation
        elif key == pygame.K_v:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.handle_scroll(-1, section="decay_chain")
        elif key == pygame.K_b:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.handle_scroll(1, section="decay_chain")
        elif key == pygame.K_c:
            if hasattr(self.renderer, 'decay_chain_scroll'):
                self.renderer.decay_chain_scroll = 0

        # Add a key to toggle all section visibility
        elif key == pygame.K_h:
            # Toggle visibility of all sections at once
            all_visible = all(self.renderer.section_visible.values())
            for section in self.renderer.section_visible:
                self.renderer.section_visible[section] = not all_visible
            logger.info(f"{'Hidden' if all_visible else 'Showing'} all info sections")
                
    def get_element_symbol(self, atomic_number):
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
        abs_seconds = abs(seconds)
        if abs_seconds == 0:
            return "0 s"
        elif abs_seconds < 1e-15:
            return f"{seconds * 1e18:.2f} as"
        elif abs_seconds < 1e-12:
            return f"{seconds * 1e15:.2f} fs"
        elif abs_seconds < 1e-9:
            return f"{seconds * 1e12:.2f} ps"
        elif abs_seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif abs_seconds < 1e-3:
            return f"{seconds * 1e6:.2f} μs"
        elif abs_seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        elif abs_seconds < 60:
            return f"{seconds:.2f} s"
        elif abs_seconds < 3600:
            return f"{seconds / 60:.2f} min"
        elif abs_seconds < 86400:
            return f"{seconds / 3600:.2f} h"
        elif abs_seconds < 31557600:
            return f"{seconds / 86400:.2f} days"
        else:
            return f"{seconds / 31557600:.2f} years"
    
    def run(self):
        logger.info("Starting simulation")
        last_time = time.time()
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0

                
                self.handle_events()
                self.update_simulation(dt)
                self.renderer.render(self.nucleus, self.particles, 
                                    self.camera_pos, self.zoom_level, 
                                    self.time_scale, self.accuracy, 
                                    self.physics_dt, self.substeps_used,
                                    self.max_substeps, self.gpu_available,
                                    self.decay_counts, self.time_passed,
                                    self.input_mode, self.input_values, self.input_cursor)
                
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


