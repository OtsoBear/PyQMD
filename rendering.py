import pygame
import math
from particles import ParticleType

class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.simulation_width = min(800, self.width - 400)  # Adjust simulation area
        self.simulation_height = self.height 
        self.font = pygame.font.SysFont('Arial', 16)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.cache = {}
        self.fm_per_unit = 0.5
        self.info_panel_scroll = 0  # Scrolling offset for info panel
        self.decay_chain_scroll = 0  # Track scrolling for decay chain
        self.max_decay_scroll = 0    # Maximum scroll value for decay chain
    
    def resize(self, width, height):
        """Update renderer dimensions after window resize"""
        self.width = width
        self.height = height
        self.simulation_width = min(800, width - 400)  # Reserve space for info panel
        self.simulation_height = height
        # Clear text cache on resize
        self.cache = {}
    
    def render(self, nucleus, particles, camera_pos, zoom, time_scale, 
              accuracy, physics_dt, substeps, max_substeps, gpu_available,
              decay_counts, time_passed):
        self.screen.fill((0, 0, 0))
        self.draw_ruler(camera_pos, zoom)
        
        # Ensure zoom is properly applied
        effective_zoom = max(0.1, zoom)  # Prevent zoom from being too small
        
        if nucleus:
            sorted_particles = sorted(nucleus.particles, key=lambda p: p.y)
            for particle in sorted_particles:
                self.draw_particle(particle, camera_pos, effective_zoom)
        
        for particle in particles:
            fade = particle.age / particle.lifetime if particle.lifetime < float('inf') else 0
            self.draw_particle(particle, camera_pos, effective_zoom, fade)
        
        self.draw_info_panel(nucleus, effective_zoom, time_scale, accuracy, physics_dt, 
                          substeps, max_substeps, gpu_available, decay_counts, time_passed)
        
        pygame.display.flip()
    
    def draw_particle(self, particle, camera_pos, zoom, fade=0):
        x, y = self.world_to_screen(particle.x, particle.y, camera_pos, zoom)
        if not (0 <= x < self.simulation_width and 0 <= y < self.simulation_height):
            return
            
        radius = max(1, int(particle.radius * zoom))
        color = particle.get_color()
        if fade > 0:
            color = tuple(int(c * (1 - min(fade, 1.0))) for c in color)
            
        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        
        if particle.type == ParticleType.PROTON and radius > 3:
            highlight_radius = max(1, int(radius * 0.3))
            highlight_offset = max(1, int(radius * 0.2))
            highlight_color = (255, 150, 150)
            if fade > 0:
                highlight_color = tuple(int(c * (1 - min(fade, 1.0))) for c in highlight_color)
            pygame.draw.circle(self.screen, highlight_color,
                             (int(x - highlight_offset), int(y - highlight_offset)),
                             highlight_radius)
        elif particle.type == ParticleType.NEUTRON and radius > 2:
            ring_color = (150, 150, 200)
            if fade > 0:
                ring_color = tuple(int(c * (1 - min(fade, 1.0))) for c in ring_color)
            pygame.draw.circle(self.screen, ring_color, (int(x), int(y)), radius - 1, 1)
    
    def draw_ruler(self, camera_pos, zoom):
        ruler_width = self.simulation_width * 0.25
        sim_units = ruler_width / zoom
        fm_length = sim_units * self.fm_per_unit
        
        nice_values = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        for val in nice_values:
            if val >= fm_length:
                nice_fm = nice_values[max(0, nice_values.index(val) - 1)]
                break
        else:
            nice_fm = nice_values[-1]
        
        ruler_length = (nice_fm / self.fm_per_unit) * zoom
        ruler_x = 50
        ruler_y = self.simulation_height - 50
        
        pygame.draw.line(self.screen, (200, 200, 200), 
                       (ruler_x, ruler_y), 
                       (ruler_x + ruler_length, ruler_y), 2)
        
        for i in range(6):
            tick_x = ruler_x + (ruler_length * i / 5)
            tick_height = 10 if i % 5 == 0 else 5
            pygame.draw.line(self.screen, (200, 200, 200),
                           (tick_x, ruler_y), 
                           (tick_x, ruler_y - tick_height), 1)
        
        text = self.font.render(f"{nice_fm} fm", True, (200, 200, 200))
        self.screen.blit(text, (ruler_x + ruler_length/2 - text.get_width()/2, ruler_y - 25))
        
        label = self.font.render("Scale: ", True, (200, 200, 200))
        self.screen.blit(label, (ruler_x - 10 - label.get_width(), ruler_y - 10))
    
    def world_to_screen(self, x, y, camera_pos, zoom):
        """Convert world coordinates to screen coordinates with proper zoom"""
        center_x = self.simulation_width / 2
        center_y = self.simulation_height / 2
        screen_x = center_x + (x - camera_pos[0]) * zoom
        screen_y = center_y + (y - camera_pos[1]) * zoom
        return screen_x, screen_y
    
    def get_text(self, key, text, color):
        cache_key = f"{key}_{text}_{color}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.font.render(text, True, color)
        return self.cache[cache_key]
    
    def draw_info_panel(self, nucleus, zoom, time_scale, accuracy, physics_dt, 
                      substeps, max_substeps, gpu_available, decay_counts, time_passed):
        # Collect all info panel items first
        self.info_items = []  # Reset items
        
        # Fixed position for info panel
        x = self.simulation_width + 20
        y = 20 - self.info_panel_scroll  # Apply scroll offset
        line_height = 25
        
        # Helper to add items to the info panel
        def add_item(key, text, color):
            self.info_items.append((key, text, color, x, y))
            return y + line_height
        
        # Add all info items
        y = add_item("accel", f"Acceleration: {'GPU' if gpu_available else 'CPU'}", 
                    (100, 255, 100) if gpu_available else (255, 100, 100))
        
        y = add_item("zoom", f"Zoom: {zoom:.1f}x", (200, 200, 255))
        
        if nucleus:
            p, n = nucleus.protons, nucleus.neutrons
            element_name, symbol = self.get_element_name(p)
            y = add_item("element", f"Element: {element_name} ({symbol})", (255, 255, 255))
            
            mass = p + n
            y = add_item("mass", f"Isotope: {symbol}-{mass}", (255, 255, 255))
            
            y = add_item("protons", f"Protons: {p}", (255, 100, 100))
            
            y = add_item("neutrons", f"Neutrons: {n}", (100, 100, 255))
            
            half_life = nucleus.stability
            if half_life == float('inf'):
                stability_text = "Stable"
                stability_color = (100, 255, 100)
            elif half_life > 31557600000000.0:  # > 1M years
                stability_text = f"{half_life/31557600000000.0:.2e} million years"
                stability_color = (100, 255, 100)
            elif half_life > 31557600000.0:  # > 1K years
                stability_text = f"{half_life/31557600000.0:.2e} millennia"
                stability_color = (180, 255, 100)
            elif half_life > 31557600.0:  # > 1 year
                stability_text = f"{half_life/31557600.0:.2f} years"
                stability_color = (255, 255, 0)
            elif half_life > 86400.0:  # > 1 day
                stability_text = f"{half_life/86400.0:.2f} days"
                stability_color = (255, 150, 0)
            elif half_life > 3600.0:
                stability_text = f"{half_life/3600.0:.2f} hours"
                stability_color = (255, 100, 0)
            else:
                stability_text = f"{half_life:.2f} seconds"
                stability_color = (255, 80, 80)
                
            y = add_item("halflife", f"Half-life: {stability_text}", stability_color)
            
            y += line_height
            y = add_item("decays", "Decay Statistics:", (255, 255, 255))
            
            colors = {
                "ALPHA": (255, 200, 0),
                "BETA_MINUS": (0, 255, 255),
                "BETA_PLUS": (255, 0, 255),
                "GAMMA": (0, 255, 0),
                "NEUTRON_EMISSION": (100, 100, 255),
                "PROTON_EMISSION": (255, 100, 100),
                "SPONTANEOUS_FISSION": (255, 128, 0),
            }
            
            for decay_type, count in decay_counts.items():
                if count > 0:
                    y = add_item(f"decay_{decay_type}", 
                               f"{decay_type}: {count}", 
                               colors.get(decay_type, (200, 200, 200)))
        
        y += line_height
        
        # Show simulation time with appropriate units
        time_value, time_unit = self.format_time_value(time_passed)
        y = add_item("simtime", f"Simulation Time: {time_value:.2f} {time_unit}", (255, 255, 255))
        
        # Show time scale with appropriate units
        time_text = self.format_time_scale(time_scale)
        y = add_item("time", f"Time Scale: {time_text}", (255, 255, 255))
        
        if substeps > 0:
            ratio = substeps / max_substeps if max_substeps > 0 else 0
            color = (255, 100, 100) if ratio > 0.95 else (255, 200, 100) if ratio > 0.75 else (100, 255, 100)
            y = add_item("substeps", f"Physics substeps: {substeps}/{max_substeps} ({ratio:.0%})", color)
            
            y = add_item("dt", f"Physics dt: {physics_dt:.6f}s", (200, 200, 255))
            
            y = add_item("accuracy", f"Physics accuracy: {accuracy:.2f}", (200, 200, 255))
        
        # Draw decay chain if nucleus exists and has actual decays
        if nucleus and hasattr(nucleus, 'decay_chain') and len(nucleus.decay_chain) > 1:
            y += line_height
            self.screen.blit(self.get_text("decay_chain_title", "Decay Chain:", (255, 220, 150)), (x, y))
            y += line_height
            
            # Show the complete chain, not just the last 5 items
            # Skip the initial state (index 0) which isn't a real decay
            full_decay_chain = nucleus.decay_chain[1:] if len(nucleus.decay_chain) > 1 else []
            
            # Apply scroll offset if chain is too long
            chain_area_height = self.height - y - 200  # Reserve space for other info
            visible_items = max(1, int(chain_area_height / line_height))
            
            # Calculate max scroll based on chain length
            self.max_decay_scroll = max(0, len(full_decay_chain) - visible_items)
            
            # Apply scrolling to select which items to show
            scroll_start = min(self.decay_chain_scroll, self.max_decay_scroll)
            end_idx = min(len(full_decay_chain), scroll_start + visible_items)
            
            # Show scrolling indicators if needed
            if self.max_decay_scroll > 0:
                # Show up/down indicators if there are more items
                if scroll_start > 0:
                    self.screen.blit(self.get_text("scroll_up", "↑ More ↑", (180, 180, 180)), (x + 50, y - line_height))
                if scroll_start < self.max_decay_scroll:
                    # Calculate position for the "more" indicator at bottom of visible area
                    bottom_y = y + (visible_items * line_height)
                    self.screen.blit(self.get_text("scroll_down", "↓ More ↓", (180, 180, 180)), (x + 50, bottom_y))
            
            # Get the visible portion of the chain based on scroll position
            display_chain = full_decay_chain[scroll_start:end_idx]
                
            # Draw numbering to show progress through the chain
            chain_header = f"Decays ({scroll_start+1}-{end_idx} of {len(full_decay_chain)})"
            self.screen.blit(self.get_text("chain_count", chain_header, (220, 220, 220)), (x, y))
            y += line_height
            
            # Draw each step in the chain
            for i, step in enumerate(display_chain):
                try:
                    # Check if the step tuple includes decay time (6 elements)
                    if len(step) >= 6:
                        orig_element, orig_mass, decay_type, new_element, new_mass, decay_time = step
                    else:
                        # Handle older format tuples with 5 elements (no decay time)
                        orig_element, orig_mass, decay_type, new_element, new_mass = step
                        decay_time = 0
                    
                    # Ensure proper formatting by converting to strings
                    orig_element = str(orig_element)
                    orig_mass = str(orig_mass)
                    decay_type = str(decay_type)
                    new_element = str(new_element)
                    new_mass = str(new_mass)
                    
                    # Fix common display issues with decay symbols
                    if decay_type == "a":
                        decay_type = "α"
                    elif decay_type == "b-":
                        decay_type = "β-"
                    elif decay_type == "b+":
                        decay_type = "β+"
                    elif decay_type == "g":
                        decay_type = "γ"
                    
                    # Highlight current isotope
                    is_current = (i == len(display_chain) - 1)
                    color = (255, 255, 100) if is_current else (200, 200, 200)
                    
                    # Format the decay time with appropriate units
                    # Always show a decay time, even if it's very small
                    if decay_time == 0:
                        # For zero decay times (initial state or extremely fast decays),
                        # show a dash or indicate instantaneous
                        time_str = " [initial]" if i == 0 else " [<1 fs]"
                    else:
                        # Get time with appropriate units
                        time_str = " [" + self.format_time_value_with_unit(decay_time) + "]"
                    
                    # Format as: U-238 → Th-234 (α) [10.5 s]
                    decay_text = f"{scroll_start+i+1}. {orig_element}-{orig_mass} → {new_element}-{new_mass} ({decay_type}){time_str}"
                    
                    self.screen.blit(self.get_text(f"decay_step_{scroll_start+i}", decay_text, color), (x, y))
                    y += line_height
                except Exception as e:
                    # If there's an error in formatting, show a debug version
                    self.screen.blit(self.get_text(f"decay_step_error_{i}", 
                                                f"Error displaying decay step: {str(step)}", 
                                                (255, 100, 100)), (x, y))
                    y += line_height
            
        y += line_height
        
        # Controls section
        y = add_item("controls0", "Controls:", (255, 255, 150))
        y = add_item("controls1", "WASD: Move camera", (200, 200, 200))
        y = add_item("controls2", "Q/E: Zoom in/out", (200, 200, 200))
        y = add_item("controls3", "↑/↓: Change time scale by 10x", (200, 200, 200))
        y = add_item("controls4", "←/→: Fine tune time scale", (200, 200, 200))
        y = add_item("controls5", "F: Toggle auto-substeps", (200, 200, 200))
        y = add_item("controls6", "-/+: Adjust max substeps", (200, 200, 200))
        y = add_item("controls7", ",/.: Adjust physics timestep", (200, 200, 200))
        y = add_item("controls8", "SPACE: Force decay", (200, 200, 200))
        y = add_item("controls9", "1-9: Select isotopes", (200, 200, 200))
        y = add_item("controls10", "R/T/H/J/Y/B: Time presets", (200, 200, 200))
        y = add_item("controls11", "Scroll wheel: Zoom", (200, 200, 200))
        y = add_item("controls_pgupdn", "PgUp/PgDn: Scroll decay chain", (200, 200, 200))
        y = add_item("controls_reset_chain", "C: Reset decay chain scroll", (200, 200, 200))
        
        # Calculate maximum scroll value
        max_content_height = y + self.info_panel_scroll  # Total content height
        visible_height = self.height - 40  # Visible area height with padding
        self.max_scroll = max(0, max_content_height - visible_height)
        
        # Draw scroll indicators if needed
        if self.max_scroll > 0:
            # Draw scrollbar background
            scrollbar_x = self.width - 15
            scrollbar_height = self.height - 40
            pygame.draw.rect(self.screen, (50, 50, 50), 
                          (scrollbar_x, 20, 10, scrollbar_height))
            
            # Draw scroll handle
            scroll_ratio = self.info_panel_scroll / self.max_scroll
            handle_height = max(30, scrollbar_height * visible_height / max_content_height)
            handle_y = 20 + scroll_ratio * (scrollbar_height - handle_height)
            pygame.draw.rect(self.screen, (150, 150, 150), 
                          (scrollbar_x, handle_y, 10, handle_height))
            
            # Draw indicators
            if self.info_panel_scroll > 0:
                pygame.draw.polygon(self.screen, (200, 200, 200), 
                                 [(self.width-20, 15), (self.width-10, 15), (self.width-15, 5)])
            
            if self.info_panel_scroll < self.max_scroll:
                pygame.draw.polygon(self.screen, (200, 200, 200), 
                                 [(self.width-20, self.height-15), (self.width-10, self.height-15), 
                                  (self.width-15, self.height-5)])
        
        # Now render all visible items
        for key, text, color, item_x, item_y in self.info_items:
            if 10 <= item_y <= self.height - 10:  # Only draw if in visible area
                self.screen.blit(self.get_text(key, text, color), (item_x, item_y))
    
    def format_time_scale(self, time_scale):
        """Format time scale with clear, human-readable units"""
        if time_scale == 1.0:
            return "x1.0 (real-time)"
        elif time_scale > 1.0:
            if time_scale >= 31557600000000000.0:  # Billion years/s
                return f"{time_scale/31557600000000000.0:.1f} billion years/s"
            elif time_scale >= 31557600000000.0:  # Million years/s
                return f"{time_scale/31557600000000.0:.1f} million years/s"
            elif time_scale >= 31557600000.0:  # Thousand years/s
                return f"{time_scale/31557600000.0:.1f} millennia/s"
            elif time_scale >= 31557600.0: # Year/s
                return f"{time_scale/31557600.0:.1f} years/s"
            elif time_scale >= 86400.0:  # Day/s
                return f"{time_scale/86400.0:.1f} days/s"
            elif time_scale >= 3600.0:  # Hour/s
                return f"{time_scale/3600.0:.1f} hours/s"
            elif time_scale >= 60.0:  # Minute/s
                return f"{time_scale/60.0:.1f} min/s"
            else:
                return f"x{time_scale:.1f}"
        else:  # Slow motion - fix for values less than 1
            if time_scale <= 1e-15:
                return f"{time_scale/1e-18:.3g} as/s"  # attoseconds
            elif time_scale <= 1e-12:
                return f"{time_scale/1e-15:.3g} fs/s"  # femtoseconds
            elif time_scale <= 1e-9:
                return f"{time_scale/1e-12:.3g} ps/s"  # picoseconds
            elif time_scale <= 1e-6:
                return f"{time_scale/1e-9:.3g} ns/s"   # nanoseconds
            elif time_scale <= 1e-3:
                return f"{time_scale/1e-6:.3g} μs/s"   # microseconds
            elif time_scale < 1:
                return f"{time_scale*1000:.3g} ms/s"   # milliseconds
            else:
                return f"x{time_scale:.3g}"
    
    def format_time_value(self, seconds):
        """Convert simulation time to appropriate units"""
        if seconds < 60:
            return seconds, "seconds"
        elif seconds < 3600:
            return seconds / 60, "minutes"
        elif seconds < 86400:
            return seconds / 3600, "hours"
        elif seconds < 2592000:  # ~30 days
            return seconds / 86400, "days"
        elif seconds < 31557600:  # ~365.25 days
            return seconds / 2592000, "months"
        elif seconds < 31557600000:  # 1000 years
            return seconds / 31557600, "years"
        elif seconds < 31557600000000:  # 1M years
            return seconds / 31557600000, "millennia"
        else:
            return seconds / 31557600000000, "million years"

    def format_time_value_with_unit(self, seconds):
        """Format a time value with appropriate units based on scale, ensuring all times are meaningful"""
        abs_seconds = abs(seconds)
        
        if abs_seconds == 0:
            return "initial"  # For the initial state
        elif abs_seconds < 1e-15:
            # For extremely small times, still show with attoseconds
            return f"{max(0.01, seconds * 1e18):.2f} as"
        elif abs_seconds < 1e-12:
            return f"{seconds * 1e15:.2f} fs" 
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

    def get_element_name(self, atomic_number):
        elements = {
            1: ("Hydrogen", "H"), 2: ("Helium", "He"), 3: ("Lithium", "Li"),
            4: ("Beryllium", "Be"), 5: ("Boron", "B"), 6: ("Carbon", "C"),
            7: ("Nitrogen", "N"), 8: ("Oxygen", "O"), 9: ("Fluorine", "F"),
            10: ("Neon", "Ne"), 11: ("Sodium", "Na"), 12: ("Magnesium", "Mg"),
            13: ("Aluminum", "Al"), 14: ("Silicon", "Si"), 15: ("Phosphorus", "P"),
            16: ("Sulfur", "S"), 17: ("Chlorine", "Cl"), 18: ("Argon", "Ar"),
            19: ("Potassium", "K"), 20: ("Calcium", "Ca"), 21: ("Scandium", "Sc"),
            22: ("Titanium", "Ti"), 23: ("Vanadium", "V"), 24: ("Chromium", "Cr"),
            25: ("Manganese", "Mn"), 26: ("Iron", "Fe"), 27: ("Cobalt", "Co"),
            28: ("Nickel", "Ni"), 29: ("Copper", "Cu"), 30: ("Zinc", "Zn"),
            31: ("Gallium", "Ga"), 32: ("Germanium", "Ge"), 33: ("Arsenic", "As"),
            34: ("Selenium", "Se"), 35: ("Bromine", "Br"), 36: ("Krypton", "Kr"),
            37: ("Rubidium", "Rb"), 38: ("Strontium", "Sr"), 39: ("Yttrium", "Y"),
            40: ("Zirconium", "Zr"), 41: ("Niobium", "Nb"), 42: ("Molybdenum", "Mo"),
            43: ("Technetium", "Tc"), 44: ("Ruthenium", "Ru"), 45: ("Rhodium", "Rh"),
            46: ("Palladium", "Pd"), 47: ("Silver", "Ag"), 48: ("Cadmium", "Cd"),
            49: ("Indium", "In"), 50: ("Tin", "Sn"), 51: ("Antimony", "Sb"),
            52: ("Tellurium", "Te"), 53: ("Iodine", "I"), 54: ("Xenon", "Xe"),
            55: ("Cesium", "Cs"), 56: ("Barium", "Ba"), 57: ("Lanthanum", "La"),
            58: ("Cerium", "Ce"), 59: ("Praseodymium", "Pr"), 60: ("Neodymium", "Nd"),
            61: ("Promethium", "Pm"), 62: ("Samarium", "Sm"), 63: ("Europium", "Eu"),
            64: ("Gadolinium", "Gd"), 65: ("Terbium", "Tb"), 66: ("Dysprosium", "Dy"),
            67: ("Holmium", "Ho"), 68: ("Erbium", "Er"), 69: ("Thulium", "Tm"),
            70: ("Ytterbium", "Yb"), 71: ("Lutetium", "Lu"), 72: ("Hafnium", "Hf"),
            73: ("Tantalum", "Ta"), 74: ("Tungsten", "W"), 75: ("Rhenium", "Re"),
            76: ("Osmium", "Os"), 77: ("Iridium", "Ir"), 78: ("Platinum", "Pt"),
            79: ("Gold", "Au"), 80: ("Mercury", "Hg"), 81: ("Thallium", "Tl"),
            82: ("Lead", "Pb"), 83: ("Bismuth", "Bi"), 84: ("Polonium", "Po"),
            85: ("Astatine", "At"), 86: ("Radon", "Rn"), 87: ("Francium", "Fr"),
            88: ("Radium", "Ra"), 89: ("Actinium", "Ac"), 90: ("Thorium", "Th"),
            91: ("Protactinium", "Pa"), 92: ("Uranium", "U"), 93: ("Neptunium", "Np"),
            94: ("Plutonium", "Pu"), 95: ("Americium", "Am"), 96: ("Curium", "Cm"),
            97: ("Berkelium", "Bk"), 98: ("Californium", "Cf"), 99: ("Einsteinium", "Es"),
            100: ("Fermium", "Fm"), 101: ("Mendelevium", "Md"), 102: ("Nobelium", "No"),
            103: ("Lawrencium", "Lr"), 104: ("Rutherfordium", "Rf"), 105: ("Dubnium", "Db"),
            106: ("Seaborgium", "Sg"), 107: ("Bohrium", "Bh"), 108: ("Hassium", "Hs"),
            109: ("Meitnerium", "Mt"), 110: ("Darmstadtium", "Ds"), 111: ("Roentgenium", "Rg"),
            112: ("Copernicium", "Cn"), 113: ("Nihonium", "Nh"), 114: ("Flerovium", "Fl"),
            115: ("Moscovium", "Mc"), 116: ("Livermorium", "Lv"), 117: ("Tennessine", "Ts"),
            118: ("Oganesson", "Og")
        }
        
        if atomic_number in elements:
            return elements[atomic_number]
        return f"Element-{atomic_number}", f"E{atomic_number}"
    
    def handle_scroll(self, amount, section=None):
        """Handle scrolling for different panel sections"""
        if section == "decay_chain":
            self.decay_chain_scroll = max(0, min(self.max_decay_scroll, 
                                                self.decay_chain_scroll + amount))
        else:
            # Default info panel scrolling
            self.info_panel_scroll = max(0, self.info_panel_scroll + amount)
