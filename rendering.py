import pygame
import math
from particles import ParticleType

class Renderer:
    def __init__(self, screen):
        pygame.font.init()
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.simulation_width = min(800, self.width - 550)
        self.simulation_height = self.height 
        
        # Modern fonts for better typography
        try:
            self.font = pygame.font.SysFont('Segoe UI', 16)
            self.small_font = pygame.font.SysFont('Segoe UI', 14)
            self.title_font = pygame.font.SysFont('Segoe UI', 20, bold=True)
            self.header_font = pygame.font.SysFont('Segoe UI', 18, bold=True)
        except:
            self.font = pygame.font.SysFont('Arial', 16)
            self.small_font = pygame.font.SysFont('Arial', 14)
            self.title_font = pygame.font.SysFont('Arial', 20, bold=True)
            self.header_font = pygame.font.SysFont('Arial', 18, bold=True)
            
        self.cache = {}
        self.fm_per_unit = 0.5
        self.info_panel_scroll = 0
        self.decay_chain_scroll = 0
        self.max_decay_scroll = 0
        
        # Grid settings - physical scale
        self.fm_per_grid = 2.0  # Each grid cell = 2 femtometers
        
        # Section visibility flags
        self.section_visible = {
            "nucleus": True,
            "controls": True,
            "physics": True
        }
        
        # Track which section is being hovered for toggle buttons
        self.hovered_section = None
        
        # Modern dark theme color palette (VS Code/Spotify inspired)
        self.colors = {
            'bg': (18, 18, 24),              # Background
            'panel_bg': (30, 30, 38),        # Panel background
            'panel_header': (40, 42, 54),    # Panel header background
            'border': (60, 60, 70),          # Border color
            'text': (220, 220, 230),         # Regular text
            'dim_text': (160, 160, 170),     # Dimmed text for less important info
            'highlight': (86, 156, 214),     # Highlighted text (VS Code blue)
            'title': (214, 222, 235),        # Title text
            'success': (80, 200, 120),       # Success/good status
            'warning': (240, 180, 60),       # Warning status
            'error': (240, 80, 80),          # Error status
            'accent1': (197, 134, 192),      # Accent color 1 (purple)
            'accent2': (86, 182, 194),       # Accent color 2 (teal)
            'grid': (35, 35, 45),            # Grid lines color
            'ruler': (70, 70, 90),           # Ruler color
            'active': (45, 45, 60),          # Active element
            'inactive': (38, 38, 48),        # Inactive element
            'selection': (55, 100, 150),     # Selection highlight
            'tab_active': (50, 100, 150),    # Active tab
            'tab_inactive': (40, 45, 55),    # Inactive tab
        }
        
    def resize(self, width, height):
        self.width = width
        self.height = height
        self.simulation_width = min(800, width - 550)
        self.simulation_height = height
        self.cache = {}
        self.info_panel_scroll = 0
        self.decay_chain_scroll = 0
    
    def render(self, nucleus, particles, camera_pos, zoom, time_scale, 
              accuracy, physics_dt, substeps, max_substeps, gpu_available,
              decay_counts, time_passed, input_mode=False, input_values=None, input_cursor=0):
        # Fill with the dark background
        self.screen.fill(self.colors['bg'])
        
        # Draw the simulation grid for better spatial awareness
        self.draw_simulation_background(camera_pos, zoom)
        self.draw_ruler(camera_pos, zoom)
        
        # Draw simulation elements
        effective_zoom = max(0.1, zoom)
        
        if nucleus:
            sorted_particles = sorted(nucleus.particles, key=lambda p: p.y)
            for particle in sorted_particles:
                self.draw_particle(particle, camera_pos, effective_zoom)
        
        for particle in particles:
            fade = particle.age / particle.lifetime if particle.lifetime < float('inf') else 0
            self.draw_particle(particle, camera_pos, effective_zoom, fade)
        
        # Draw information panels
        self.draw_info_panel(nucleus, zoom, time_scale, accuracy, physics_dt, 
                          substeps, max_substeps, gpu_available, decay_counts, time_passed)
        
        # Draw decay chain panel if applicable
        if nucleus and hasattr(nucleus, 'decay_chain') and len(nucleus.decay_chain) > 1:
            self.draw_decay_chain(nucleus)
        
        # Draw input panel when in input mode
        if input_mode and input_values is not None:
            self.draw_input_panel(input_values, input_cursor)
        
        pygame.display.flip()
    
    def draw_simulation_background(self, camera_pos, zoom):
        # Scale grid to represent fixed physical units (femtometers)
        # Convert from femtometers to simulation units and apply zoom
        grid_size = (self.fm_per_grid / self.fm_per_unit) * zoom
        
        # Ensure grid size is reasonable for display purposes
        grid_size = max(5, min(100, grid_size))
        
        # Calculate visible grid lines based on camera position and zoom
        offset_x = (camera_pos[0] * zoom) % grid_size
        offset_y = (camera_pos[1] * zoom) % grid_size
        
        # Draw grid lines with physical scaling
        grid_color = self.colors['grid']
        
        # Draw vertical grid lines
        x = -offset_x
        while x < self.simulation_width:
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.simulation_height), 1)
            x += grid_size
            
        # Draw horizontal grid lines
        y = -offset_y
        while y < self.simulation_height:
            pygame.draw.line(self.screen, grid_color, (0, y), (self.simulation_width, y), 1)
            y += grid_size
            
        # Draw grid scale indicator in corner
        scale_text = f"Grid: {self.fm_per_grid} fm"
        scale_surf = self.get_text("grid_scale", scale_text, self.colors['dim_text'])
        self.screen.blit(scale_surf, (10, 10))
    
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
        
        # Draw ruler background
        ruler_bg_rect = (ruler_x - 10, ruler_y - 20, ruler_length + 20, 35)
        pygame.draw.rect(self.screen, self.colors['panel_bg'], ruler_bg_rect)
        pygame.draw.rect(self.screen, self.colors['border'], ruler_bg_rect, 1)
        
        # Draw ruler line
        pygame.draw.line(self.screen, self.colors['ruler'], 
                       (ruler_x, ruler_y), 
                       (ruler_x + ruler_length, ruler_y), 2)
        
        # Draw tick marks
        for i in range(6):
            tick_x = ruler_x + (ruler_length * i / 5)
            tick_height = 10 if i % 5 == 0 else 5
            pygame.draw.line(self.screen, self.colors['ruler'],
                           (tick_x, ruler_y), 
                           (tick_x, ruler_y - tick_height), 1)
        
        # Draw scale label with modern styling
        text = self.font.render(f"{nice_fm} fm", True, self.colors['title'])
        self.screen.blit(text, (ruler_x + ruler_length/2 - text.get_width()/2, ruler_y - 25))
    
    def world_to_screen(self, x, y, camera_pos, zoom):
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
    
    def get_title_text(self, text, color):
        cache_key = f"title_{text}_{color}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.title_font.render(text, True, color)
        return self.cache[cache_key]
    
    def get_header_text(self, text, color):
        cache_key = f"header_{text}_{color}"
        if cache_key not in self.cache:
            self.cache[cache_key] = self.header_font.render(text, True, color)
        return self.cache[cache_key]
    
    def draw_info_panel(self, nucleus, zoom, time_scale, accuracy, physics_dt, 
                      substeps, max_substeps, gpu_available, decay_counts, time_passed):
        panel_x = self.simulation_width + 10
        panel_width = min(300, self.width - self.simulation_width - 330)
        panel_height = self.height - 20
        
        # Draw panel background with modern styling
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                       (panel_x, 10, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (panel_x, 10, panel_width, panel_height), 1)
        
        # Header area with slight background difference
        header_height = 40
        pygame.draw.rect(self.screen, self.colors['panel_header'], 
                       (panel_x, 10, panel_width, header_height))
        
        # Draw title
        title = "Nuclear Physics Simulator"
        title_text = self.get_title_text(title, self.colors['title'])
        self.screen.blit(title_text, 
                       (panel_x + (panel_width - title_text.get_width()) // 2, 20))
        
        # Draw sections one after another vertically
        current_y = header_height + 15
        content_x = panel_x + 15
        
        # Draw nucleus info section if visible
        current_y = self.draw_section_header("Nucleus Info", "nucleus", 
                                         panel_x, current_y, panel_width)
        if self.section_visible["nucleus"]:
            current_y = self.draw_nucleus_info_section(content_x, current_y, panel_width, 
                                                  nucleus, time_scale, time_passed, 
                                                  decay_counts, gpu_available)
        
        # Add spacing between sections
        current_y += 10
        
        # Draw controls section if visible
        current_y = self.draw_section_header("Controls", "controls", 
                                         panel_x, current_y, panel_width)
        if self.section_visible["controls"]:
            current_y = self.draw_controls_section(content_x, current_y, panel_width)
        
        # Add spacing between sections
        current_y += 10
        
        # Draw physics section if visible
        current_y = self.draw_section_header("Physics Parameters", "physics", 
                                         panel_x, current_y, panel_width)
        if self.section_visible["physics"]:
            current_y = self.draw_physics_section(content_x, current_y, panel_width, 
                                             zoom, accuracy, physics_dt, substeps, max_substeps)
    
    def draw_section_header(self, title, section_id, x, y, width):
        """Draw a collapsible section header with toggle button"""
        header_height = 30
        header_color = self.colors['panel_header']
        
        # Draw section header background
        pygame.draw.rect(self.screen, header_color, (x, y, width, header_height))
        pygame.draw.line(self.screen, self.colors['border'], 
                       (x, y + header_height), 
                       (x + width, y + header_height), 1)
        
        # Draw section title
        header_text = self.get_header_text(title, self.colors['highlight'])
        self.screen.blit(header_text, (x + 15, y + (header_height - header_text.get_height()) // 2))
        
        # Draw toggle button ([+] or [-])
        toggle_x = x + width - 30
        toggle_y = y + (header_height - 20) // 2
        toggle_width = 20
        toggle_height = 20
        
        # Check if mouse is hovering over the toggle button
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = (toggle_x <= mouse_pos[0] <= toggle_x + toggle_width and
                     toggle_y <= mouse_pos[1] <= toggle_y + toggle_height)
        
        # If mouse is being clicked on this button, store which section is being toggled
        if is_hovered and pygame.mouse.get_pressed()[0]:
            self.hovered_section = section_id
        
        if is_hovered:
            # Draw highlighted button
            pygame.draw.rect(self.screen, self.colors['selection'], 
                           (toggle_x, toggle_y, toggle_width, toggle_height))
        else:
            # Draw normal button
            pygame.draw.rect(self.screen, self.colors['tab_inactive'], 
                           (toggle_x, toggle_y, toggle_width, toggle_height))
        
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (toggle_x, toggle_y, toggle_width, toggle_height), 1)
        
        # Draw + or - symbol based on visibility
        toggle_symbol = "-" if self.section_visible[section_id] else "+"
        symbol_text = self.get_text(f"toggle_{section_id}", toggle_symbol, self.colors['text'])
        self.screen.blit(symbol_text, 
                        (toggle_x + (toggle_width - symbol_text.get_width()) // 2, 
                         toggle_y + (toggle_height - symbol_text.get_height()) // 2))
        
        return y + header_height
    
    def draw_nucleus_info_section(self, x, y, width, nucleus, time_scale, time_passed, decay_counts, gpu_available):
        """Draw nucleus information section with better horizontal spacing"""
        if not nucleus:
            # Show message if no nucleus
            no_nucleus_text = self.get_text("no_nucleus", "No nucleus created", self.colors['dim_text'])
            self.screen.blit(no_nucleus_text, (x, y + 10))
            return y + 40
        
        content_y = y + 10
        line_height = 24
        
        # More compact layout with better spacing
        p, n = nucleus.protons, nucleus.neutrons
        element_name, symbol = self.get_element_name(p)
        mass = p + n
        
        # Element header with compact display
        self.screen.blit(self.get_header_text(f"{element_name} ({symbol}-{mass})", self.colors['highlight']), 
                        (x, content_y))
        content_y += line_height + 5
        
        # For very small widths, use single column
        use_single_column = width < 200
        col_width = width - 30 if use_single_column else (width - 40) // 2
        
        # Basic nucleus properties with better spacing
        properties = [
            ("Protons:", f"{p}", (255, 100, 100)),
            ("Neutrons:", f"{n}", (100, 100, 255)),
            ("Half-life:", self.get_stability_text(nucleus.stability)[0], 
             self.get_stability_text(nucleus.stability)[1])
        ]
        
        for i, (label, value, color) in enumerate(properties):
            if use_single_column:
                # Single column layout for narrow panels
                label_text = self.get_text(f"label_{i}", label, self.colors['dim_text'])
                value_text = self.get_text(f"value_{i}", value, color)
                
                self.screen.blit(label_text, (x, content_y))
                # Make sure value doesn't overlap by positioning it with enough space
                value_x = x + label_text.get_width() + 10
                self.screen.blit(value_text, (value_x, content_y))
                content_y += line_height
            else:
                # Two column layout for wider panels
                row, col = i // 2, i % 2
                item_x = x + col * col_width
                item_y = content_y + row * line_height
                
                # Draw label and value with controlled spacing
                label_text = self.get_text(f"label_{i}", label, self.colors['dim_text'])
                value_text = self.get_text(f"value_{i}", value, color)
                
                self.screen.blit(label_text, (item_x, item_y))
                # Position value with fixed offset to avoid overlap
                value_x = item_x + 70 
                self.screen.blit(value_text, (value_x, item_y))
        
        # If using two columns, adjust content_y accordingly
        if not use_single_column:
            content_y += ((len(properties) + 1) // 2) * line_height
            
        # Time information with separator
        content_y += 5
        pygame.draw.line(self.screen, self.colors['border'], 
                       (x, content_y), (x + width - 30, content_y), 1)
        content_y += 8
        
        time_value, time_unit = self.format_time_value(time_passed)
        time_text = f"Time: {time_value:.2f} {time_unit}"
        self.screen.blit(self.get_text("time_info", time_text, self.colors['text']), 
                        (x, content_y))
        content_y += line_height
        
        scale_text = f"Scale: {self.format_time_scale(time_scale)}"
        self.screen.blit(self.get_text("scale_info", scale_text, self.colors['text']), 
                        (x, content_y))
        content_y += line_height
        
        # Decay counts with better organization
        content_y += 8
        pygame.draw.line(self.screen, self.colors['border'], 
                       (x, content_y), (x + width - 30, content_y), 1)
        content_y += 8
        
        self.screen.blit(self.get_text("decay_header", "Decay Counts:", self.colors['text']), 
                        (x, content_y))
        content_y += line_height
        
        # Show decay counts in a more compact grid
        decay_types = [
            ("α", "ALPHA", (255, 200, 0)),
            ("β-", "BETA_MINUS", (0, 255, 255)), 
            ("β+", "BETA_PLUS", (255, 0, 255)),
            ("γ", "GAMMA", (0, 255, 0)),
            ("n", "NEUTRON_EMISSION", (100, 100, 255)),
            ("p", "PROTON_EMISSION", (255, 100, 100)),
        ]
        
        # Calculate how many decay types to show per row based on width
        col_count = 3 if width >= 250 else (2 if width >= 180 else 1)
        shown = 0
        
        for i, (symbol, decay_name, color) in enumerate(decay_types):
            if decay_counts.get(decay_name, 0) > 0:
                col = shown % col_count
                row = shown // col_count
                shown += 1
                
                decay_x = x + col * ((width - 30) // col_count)
                decay_y = content_y + row * line_height
                
                count_text = f"{symbol}: {decay_counts[decay_name]}"
                self.screen.blit(self.get_text(f"decay_{decay_name}", count_text, color), 
                               (decay_x, decay_y))
        
        # Adjust content_y based on how many decay types were shown
        if shown > 0:
            content_y += (((shown - 1) // col_count) + 1) * line_height
        
        return content_y + 5
    
    def draw_controls_section(self, x, y, width):
        """Draw controls section with better layout"""
        content_y = y + 10
        line_height = 21
        
        # Calculate number of columns based on width
        col_count = 2 if width >= 250 else 1
        col_width = (width - 30) // col_count if col_count > 1 else width - 30
        
        # Group controls into logical categories
        control_groups = [
            {
                "title": "Camera Controls",
                "color": self.colors['accent1'],
                "controls": [
                    ("WASD", "Move camera"),
                    ("Q/E", "Zoom in/out"),
                    ("R", "Reset zoom"),
                    ("G", "Change grid scale")
                ]
            },
            {
                "title": "Simulation Controls",
                "color": self.colors['accent2'],
                "controls": [
                    ("UP/DOWN", "10x time scale"),
                    ("I/K", "2x time scale"),
                    ("O", "Reset time"),
                    ("SPACE", "Force decay"),
                    ("1-9", "Select isotopes"),
                    ("N", "Custom nucleus")
                ]
            },
            {
                "title": "Advanced Controls",
                "color": self.colors['highlight'],
                "controls": [
                    ("P/L", "Adjust substeps"),
                    ("X/Z", "Adjust dt"),
                    ("F", "Auto-substeps"),
                    ("V/B", "Scroll decay chain"),
                    ("C", "Reset chain scroll"),
                    ("H", "Hide/show sections")
                ]
            }
        ]
        
        # Layout groups side by side or stacked based on available width
        col_heights = [content_y] * col_count
        current_col = 0
        
        for group in control_groups:
            # Get the next column with the least height
            current_col = col_heights.index(min(col_heights))
            col_x = x + current_col * col_width
            group_y = col_heights[current_col]
            
            # Draw group title
            self.screen.blit(self.get_text(f"controls_{group['title']}", 
                                        group["title"], group["color"]),
                           (col_x, group_y))
            group_y += line_height
            
            # Draw each control in this group with better spacing
            for key, action in group["controls"]:
                key_text = self.get_text(f"key_{key}", key, self.colors['text'])
                action_text = self.get_text(f"action_{action}", action, self.colors['dim_text'])
                
                self.screen.blit(key_text, (col_x, group_y))
                # Position action text with fixed offset to avoid overlap
                action_x = col_x + max(60, key_text.get_width() + 8)
                # If action text would overflow, move to next line
                if action_x + action_text.get_width() > col_x + col_width:
                    action_x = col_x + 10
                    group_y += line_height
                    
                self.screen.blit(action_text, (action_x, group_y))
                group_y += line_height
            
            # Update column height
            col_heights[current_col] = group_y + 10
        
        return max(col_heights) + 5
    
    def draw_physics_section(self, x, y, width, zoom, accuracy, physics_dt, substeps, max_substeps):
        """Draw physics section with better layout"""
        content_y = y + 10
        line_height = 24
        
        # Determine if we use one or two columns
        use_single_column = width < 250
        col_width = width - 30 if use_single_column else (width - 40) // 2
        
        # Draw physics parameters in a grid
        params = [
            ("Zoom Level:", f"{zoom:.1f}x", self.colors['text']),
            ("Physics dt:", f"{physics_dt:.5f}s", self.colors['text']),
            ("Accuracy:", f"{accuracy:.1f}", self.colors['text']),
            ("Substeps:", f"{substeps}/{max_substeps}", self.get_substeps_color(substeps, max_substeps))
        ]
        
        for i, (label, value, color) in enumerate(params):
            if use_single_column:
                # Single column layout
                label_text = self.get_text(f"physics_label_{i}", label, self.colors['dim_text'])
                value_text = self.get_text(f"physics_value_{i}", value, color)
                
                self.screen.blit(label_text, (x, content_y))
                # Position value with enough space to avoid overlap
                value_x = x + max(90, label_text.get_width() + 10)
                self.screen.blit(value_text, (value_x, content_y))
                content_y += line_height
            else:
                # Two column layout
                row = i // 2
                col = i % 2
                param_x = x + col * col_width
                param_y = content_y + row * line_height
                
                label_text = self.get_text(f"physics_label_{i}", label, self.colors['dim_text'])
                value_text = self.get_text(f"physics_value_{i}", value, color)
                
                self.screen.blit(label_text, (param_x, param_y))
                # Position value with fixed offset
                value_x = param_x + 90
                self.screen.blit(value_text, (value_x, param_y))
        
        # Adjust content_y based on layout
        if not use_single_column:
            content_y += ((len(params) + 1) // 2) * line_height
        
        # Add explanations with separator
        content_y += 5
        pygame.draw.line(self.screen, self.colors['border'], 
                       (x, content_y), 
                       (x + width - 30, content_y), 1)
        content_y += 8
        
        explanations = [
            "Lower physics dt = more accurate but slower",
            "Higher accuracy = more precise forces",
            "Substeps adapt to time scale if auto-adjust is on",
            "Press H to toggle section visibility"
        ]
        
        # Show explanations with word wrapping for narrow panels
        for text in explanations:
            if use_single_column and len(text) > 28:  # Approximate character limit
                # Split long text into multiple lines
                words = text.split()
                line = ""
                for word in words:
                    test_line = f"{line} {word}".strip()
                    # Check if adding this word would exceed width
                    test_text = self.get_text(f"test_wrap", test_line, self.colors['dim_text'])
                    if test_text.get_width() > width - 40:
                        # Render current line
                        if line:
                            expl_text = self.get_text(f"expl_wrap", line, self.colors['dim_text'])
                            self.screen.blit(expl_text, (x, content_y))
                            content_y += line_height - 6  # Slightly reduced for wrapped text
                        line = word
                    else:
                        line = test_line
                
                # Render remaining text
                if line:
                    expl_text = self.get_text(f"expl_wrap_end", line, self.colors['dim_text'])
                    self.screen.blit(expl_text, (x, content_y))
                    content_y += line_height
            else:
                # No need for wrapping
                expl_text = self.get_text(f"expl_{text[:10]}", text, self.colors['dim_text'])
                self.screen.blit(expl_text, (x, content_y))
                content_y += line_height
        
        return content_y + 5
    
    def handle_mouse_click(self):
        """Process mouse click to toggle section visibility"""
        if self.hovered_section:
            # Toggle the clicked section
            self.section_visible[self.hovered_section] = not self.section_visible[self.hovered_section]
            self.hovered_section = None  # Reset after processing
            return True
        return False
    
    def get_substeps_color(self, substeps, max_substeps):
        """Get color for substeps based on how close to max"""
        if max_substeps == 0:
            return self.colors['text']
        
        ratio = substeps / max_substeps
        if ratio > 0.95:
            return self.colors['error']
        elif ratio > 0.75:
            return self.colors['warning']
        return self.colors['success']
    
    def get_stability_text(self, half_life):
        """Get formatted stability text and color"""
        if half_life == float('inf'):
            return "Stable", self.colors['success']
        elif half_life > 31557600000000.0:  # > 1M years
            return f"{half_life/31557600000000.0:.2e} M yr", self.colors['success']
        elif half_life > 31557600000.0:  # > 1K years
            return f"{half_life/31557600000.0:.2e} K yr", (180, 255, 100)
        elif half_life > 31557600.0:  # > 1 year
            return f"{half_life/31557600.0:.2f} yr", self.colors['warning']
        elif half_life > 86400.0:  # > 1 day
            return f"{half_life/86400.0:.2f} d", self.colors['warning']
        elif half_life > 3600.0:
            return f"{half_life/3600.0:.2f} h", self.colors['warning']
        else:
            return f"{half_life:.2f} s", self.colors['error']
    
    def draw_decay_chain(self, nucleus):
        """Draw decay chain panel with more compact layout"""
        # Position the decay chain panel on the right
        x = self.width - 320  # Right side with margin
        panel_width = 300
        
        # Draw background for the decay chain panel with modern styling
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                        (x, 10, panel_width, self.height - 20))
        pygame.draw.rect(self.screen, self.colors['border'], 
                        (x, 10, panel_width, self.height - 20), 1)
        
        # Header area with slight background difference
        header_height = 40
        pygame.draw.rect(self.screen, self.colors['panel_header'], 
                       (x, 10, panel_width, header_height))
        
        # Draw header
        y = 20
        line_height = 22  # Slightly reduced line height
        
        # Draw title
        title = "Decay Chain"
        title_text = self.get_title_text(title, self.colors['title'])
        self.screen.blit(title_text, 
                       (x + panel_width//2 - title_text.get_width()//2, y))
        y += header_height
        
        # Get decay chain (skip initial state)
        full_decay_chain = nucleus.decay_chain[1:] if len(nucleus.decay_chain) > 1 else []
        
        # If no decays yet, show current isotope
        if not full_decay_chain:
            if nucleus.decay_chain:  # Make sure there's at least an initial state
                initial = nucleus.decay_chain[0]
                element, mass = initial[0], initial[1]
                status_text = f"Current: {element}-{mass}"
                status_surf = self.get_text("current_isotope", status_text, self.colors['highlight'])
                self.screen.blit(status_surf, 
                                (x + panel_width//2 - status_surf.get_width()//2, y + 20))
            return
        
        # Show scrolling controls and count
        scroll_info = f"({self.decay_chain_scroll+1}-{min(len(full_decay_chain), self.decay_chain_scroll+6)} of {len(full_decay_chain)})"
        scroll_text = self.get_text("chain_count", scroll_info, self.colors['dim_text'])
        scroll_hint = self.get_text("scroll_hint", "V/B to scroll", self.colors['dim_text'])
        
        # Position scroll info and controls
        self.screen.blit(scroll_text, (x + 10, y + 5))
        self.screen.blit(scroll_hint, (x + panel_width - scroll_hint.get_width() - 10, y + 5))
        
        y += 30
        
        # Calculate max scroll based on chain length
        visible_items = 6  # Fixed number of visible items
        self.max_decay_scroll = max(0, len(full_decay_chain) - visible_items)
        
        # Clamp scroll value
        self.decay_chain_scroll = max(0, min(self.decay_chain_scroll, self.max_decay_scroll))
        scroll_start = self.decay_chain_scroll
        end_idx = min(len(full_decay_chain), scroll_start + visible_items)
        
        # Get the visible portion of the chain
        display_chain = full_decay_chain[scroll_start:end_idx]
        
        # Draw each decay step in more compact format
        for i, step in enumerate(display_chain):
            try:
                # Extract decay information including time
                if len(step) >= 6:
                    orig_element, orig_mass, decay_type, new_element, new_mass, decay_time = step
                else:
                    orig_element, orig_mass, decay_type, new_element, new_mass = step
                    decay_time = 0
                
                # Format decay elements
                orig_element = str(orig_element)
                orig_mass = str(orig_mass)
                decay_type = str(decay_type)
                new_element = str(new_element)
                new_mass = str(new_mass)
                
                # Fix decay symbols if needed
                if decay_type == "a": decay_type = "α"
                elif decay_type == "b-": decay_type = "β-"
                elif decay_type == "b+": decay_type = "β+"
                elif decay_type == "g": decay_type = "γ"
                
                # Create stepped background for alternating rows
                bg_color = self.colors['active'] if i % 2 == 0 else self.colors['panel_bg']
                
                # Highlight current isotope (most recent decay)
                is_current = (i == len(display_chain) - 1)
                text_color = self.colors['highlight'] if is_current else self.colors['text']
                
                # Draw item background
                item_height = line_height * 2
                pygame.draw.rect(self.screen, bg_color, 
                               (x + 5, y, panel_width - 10, item_height))
                
                # Calculate time text
                if decay_time == 0:
                    time_text = "[initial]" if i == 0 else "[<1 fs]"
                else:
                    time_text = self.format_compact_time(decay_time)
                
                # Draw simplified decay notation
                decay_text = f"{scroll_start+i+1}. {orig_element}-{orig_mass} → {new_element}-{new_mass} ({decay_type})"
                time_label = f"Time: {time_text}"
                
                # Position text
                self.screen.blit(self.get_text(f"decay_step_{scroll_start+i}", 
                                             decay_text, text_color), (x + 10, y + 3))
                self.screen.blit(self.get_text(f"decay_time_{scroll_start+i}", 
                                            time_label, self.colors['dim_text']), (x + 15, y + line_height + 2))
                
                # Update y position for next item
                y += item_height + 3
                
            except Exception as e:
                # Show error if something goes wrong
                error_text = f"Error: {str(e)[:20]}"
                self.screen.blit(self.get_text(f"decay_error_{i}", error_text, 
                                            self.colors['error']), (x + 10, y))
                y += line_height
        
        # Draw scroll indicators if needed
        if scroll_start > 0:
            self.screen.blit(self.get_text("scroll_up", "↑", self.colors['dim_text']), 
                            (x + panel_width - 25, header_height + 5))
            
        if scroll_start < self.max_decay_scroll:
            self.screen.blit(self.get_text("scroll_down", "↓", self.colors['dim_text']), 
                            (x + panel_width - 25, y - 5))
    
    def format_compact_time(self, seconds):
        """Format time value in most compact form"""
        abs_seconds = abs(seconds)
        
        if abs_seconds == 0:
            return "initial"
        elif abs_seconds < 1e-12:
            return f"{seconds * 1e15:.1f}fs" 
        elif abs_seconds < 1e-9:
            return f"{seconds * 1e12:.1f}ps"
        elif abs_seconds < 1e-6:
            return f"{seconds * 1e9:.1f}ns"
        elif abs_seconds < 1e-3:
            return f"{seconds * 1e6:.1f}μs"
        elif abs_seconds < 1:
            return f"{seconds * 1e3:.1f}ms"
        elif abs_seconds < 60:
            return f"{seconds:.1f}s"
        elif abs_seconds < 3600:
            return f"{seconds/60:.1f}min"
        elif abs_seconds < 86400:
            return f"{seconds/3600:.1f}h"
        elif abs_seconds < 31557600:
            return f"{seconds/86400:.1f}d"
        else:
            return f"{seconds/31557600:.1f}y"
    
    def draw_input_panel(self, input_values, cursor):
        """Draw panel for custom nucleus input"""
        # Center the panel
        panel_width = 350
        panel_height = 200
        x = (self.width - panel_width) // 2
        y = (self.height - panel_height) // 2
        
        # Draw semi-transparent overlay background
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], (x, y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.colors['border'], (x, y, panel_width, panel_height), 1)
        
        # Header area with slight background difference
        header_height = 40
        pygame.draw.rect(self.screen, self.colors['panel_header'], 
                       (x, y, panel_width, header_height))
        
        # Draw title
        title = "Create Custom Nucleus"
        title_text = self.get_title_text(title, self.colors['title'])
        self.screen.blit(title_text, 
                        (x + (panel_width - title_text.get_width()) // 2, y + 10))
        
        # Draw instructions
        instructions = "Enter protons and neutrons. TAB to switch, ENTER to create."
        instr_text = self.get_text("input_instructions", instructions, self.colors['dim_text'])
        self.screen.blit(instr_text, 
                        (x + (panel_width - instr_text.get_width()) // 2, y + 50))
        
        # Draw input fields
        field_width = 100
        field_height = 40
        field_x = x + (panel_width - field_width*2 - 20) // 2
        field_y = y + 85
        
        # Protons field
        proton_color = self.colors['selection'] if cursor == 0 else self.colors['inactive']
        pygame.draw.rect(self.screen, proton_color, (field_x, field_y, field_width, field_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (field_x, field_y, field_width, field_height), 1)
        
        protons_label = self.get_text("protons_label", "Protons", self.colors['text'])
        self.screen.blit(protons_label, 
                        (field_x + (field_width - protons_label.get_width()) // 2, 
                         field_y - 25))
        
        protons_text = self.font.render(str(input_values[0]), True, self.colors['text'])
        self.screen.blit(protons_text, 
                        (field_x + (field_width - protons_text.get_width()) // 2, 
                         field_y + (field_height - protons_text.get_height()) // 2))
        
        # Neutrons field
        neutrons_color = self.colors['selection'] if cursor == 1 else self.colors['inactive']
        pygame.draw.rect(self.screen, neutrons_color, 
                       (field_x + field_width + 20, field_y, field_width, field_height))
        pygame.draw.rect(self.screen, self.colors['border'], 
                       (field_x + field_width + 20, field_y, field_width, field_height), 1)
        
        neutrons_label = self.get_text("neutrons_label", "Neutrons", self.colors['text'])
        self.screen.blit(neutrons_label, 
                        (field_x + field_width + 20 + (field_width - neutrons_label.get_width()) // 2, 
                         field_y - 25))
        
        neutrons_text = self.font.render(str(input_values[1]), True, self.colors['text'])
        self.screen.blit(neutrons_text, 
                        (field_x + field_width + 20 + (field_width - neutrons_text.get_width()) // 2, 
                         field_y + (field_height - neutrons_text.get_height()) // 2))
        
        # Preview of the isotope
        if input_values[0] > 0:
            element_name, symbol = self.get_element_name(input_values[0])
            mass = input_values[0] + input_values[1]
            isotope_text = f"{element_name} ({symbol}-{mass})"
            
            preview_text = self.get_text("preview_isotope", isotope_text, self.colors['highlight'])
            self.screen.blit(preview_text, 
                           (x + (panel_width - preview_text.get_width()) // 2, 
                            field_y + field_height + 30))
            
        # Draw action hints at the bottom
        hint_y = y + panel_height - 30
        hint_text = self.get_text("input_hint", "ENTER to create   |   ESC to cancel", 
                                self.colors['dim_text'])
        self.screen.blit(hint_text, 
                        (x + (panel_width - hint_text.get_width()) // 2, hint_y))
    
    def handle_scroll(self, amount, section=None):
        """Handle scrolling for different panel sections"""
        if section == "decay_chain":
            self.decay_chain_scroll = max(0, min(self.max_decay_scroll, 
                                                self.decay_chain_scroll + amount))
        else:
            self.info_panel_scroll = max(0, self.info_panel_scroll + amount)
            
    def format_time_scale(self, time_scale):
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
        abs_seconds = abs(seconds)
        
        if abs_seconds == 0:
            return "initial"  # For the initial state
        elif abs_seconds < 1e-15:
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
