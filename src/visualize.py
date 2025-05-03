"""
Fixed visualization module with proper particle sizing and smooth movement.
"""
import numpy as np
import moderngl
import glfw
import time
import pygame  # We'll use Pygame for text rendering
from pygame import freetype
from quark import QuarkColor, QuarkFlavor
from baryon import BaryonType

# CRITICAL: Use much simpler shaders to avoid potential issues
VERTEX_SHADER = """
#version 330

// Input attributes
in vec2 in_pos;     // Position in NDC coordinates
in vec3 in_color;   // RGB color
in float in_size;   // Point size

// Output to fragment shader
out vec3 color;

void main() {
    // Just pass the position through
    gl_Position = vec4(in_pos, 0.0, 1.0);
    
    // Set point size directly - will be a large value
    gl_PointSize = in_size;
    
    // Pass color to fragment shader
    color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330

in vec3 color;
out vec4 fragColor;

void main() {
    // Create a circle by calculating distance from center
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center) * 2.0;
    
    // Discard pixels outside circle
    if (dist > 1.0) {
        discard;
    }
    
    // Output color with alpha
    fragColor = vec4(color, 1.0);
}
"""

# DRAMATICALLY larger base sizes
PARTICLE_SIZES = {
    'QUARK': 30.0,     # Much larger base size
    'BARYON': 60.0,    # Much larger base size
    'DEBUG': 40.0      # Debug particle size
}

# Standard colors
COLORS = {
    'UP_RED': (1.0, 0.2, 0.2),
    'UP_GREEN': (0.2, 1.0, 0.2),
    'UP_BLUE': (0.2, 0.2, 1.0),
    'DOWN_RED': (0.8, 0.0, 0.0),
    'DOWN_GREEN': (0.0, 0.8, 0.0),
    'DOWN_BLUE': (0.0, 0.0, 0.8),
    'PROTON': (1.0, 0.8, 0.0),
    'NEUTRON': (0.7, 0.7, 0.7),
    'DELTA_PP': (1.0, 0.2, 1.0),
    'DELTA_M': (0.0, 0.7, 1.0),
    'DEBUG': (1.0, 0.0, 1.0)
}

# Add more descriptive color names and explanations
PARTICLE_TYPES = {
    'UP_RED': ("Red up", "u quark (2/3 charge)"),
    'UP_GREEN': ("Green up", "u quark (2/3 charge)"),
    'UP_BLUE': ("Blue up", "u quark (2/3 charge)"),
    'DOWN_RED': ("Red down", "d quark (-1/3 charge)"),
    'DOWN_GREEN': ("Green down", "d quark (-1/3 charge)"),
    'DOWN_BLUE': ("Blue down", "d quark (-1/3 charge)"),
    'PROTON': ("Proton", "uud (charge +1)"),
    'NEUTRON': ("Neutron", "udd (charge 0)"),
    'DELTA_PP': ("Δ++", "uuu (charge +2)"),
    'DELTA_M': ("Δ-", "ddd (charge -1)")
}

class Visualizer:
    """Fixed visualization class with proper particle sizing."""
    
    def __init__(self, simulation, width=800, height=600):
        self.sim = simulation
        self.width = width
        self.height = height
        
        # Camera settings
        self.center_x = self.sim.box_size / 2  # Center position in sim coordinates
        self.center_y = self.sim.box_size / 2
        self.zoom = 10.0  # Higher zoom = larger view
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # Input tracking
        self.keys = {}
        
        # Debug mode
        self.show_debug = True
        
        # Initialize graphics
        self._init_glfw()
        self._init_gl()
        
        # Initialize pygame for text rendering
        pygame.init()
        pygame.freetype.init()
        self.font = pygame.freetype.SysFont('Arial', 14)
        self.show_labels = True
        self.show_legend = True
        
        print("FIXED visualization system initialized")
        print("Controls: Arrow keys to pan, +/- to zoom, Home to reset view, D to toggle debug, L to toggle labels, TAB to toggle legend")
    
    def _init_glfw(self):
        """Initialize GLFW window."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        
        # Create window
        self.window = glfw.create_window(
            self.width, self.height, 
            "PyQMD - FIXED VISUALIZATION", 
            None, None
        )
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        
        # Set window as current context
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_size_callback(self.window, self._resize_callback)
    
    def _init_gl(self):
        """Initialize ModernGL and prepare for rendering."""
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        
        # CRITICAL: Enable program point size
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Create shader program
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        
        # Create vertex buffer for particles
        self.max_particles = self.sim.num_particles * 2 + 10  # Extra for debug
        self.vertex_data = np.zeros(self.max_particles, dtype=[
            ('in_pos', np.float32, 2),
            ('in_color', np.float32, 3),
            ('in_size', np.float32)
        ])
        
        # Create buffer and VAO
        self.vbo = self.ctx.buffer(self.vertex_data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, 
            [(self.vbo, '2f 3f 1f', 'in_pos', 'in_color', 'in_size')]
        )
        
        # Add debug particles
        self._add_debug_particles()
    
    def _add_debug_particles(self):
        """Add debug particles at fixed positions."""
        if not self.show_debug:
            return
            
        # Center of screen
        self.vertex_data[0]['in_pos'] = (0.0, 0.0)
        self.vertex_data[0]['in_color'] = COLORS['DEBUG']
        self.vertex_data[0]['in_size'] = PARTICLE_SIZES['DEBUG']
        
        # Four corners
        self.vertex_data[1]['in_pos'] = (-0.9, -0.9)
        self.vertex_data[1]['in_color'] = (1.0, 0.0, 0.0)  # Red
        self.vertex_data[1]['in_size'] = PARTICLE_SIZES['DEBUG']
        
        self.vertex_data[2]['in_pos'] = (0.9, -0.9)
        self.vertex_data[2]['in_color'] = (0.0, 1.0, 0.0)  # Green
        self.vertex_data[2]['in_size'] = PARTICLE_SIZES['DEBUG']
        
        self.vertex_data[3]['in_pos'] = (-0.9, 0.9)
        self.vertex_data[3]['in_color'] = (0.0, 0.0, 1.0)  # Blue
        self.vertex_data[3]['in_size'] = PARTICLE_SIZES['DEBUG']
        
        self.vertex_data[4]['in_pos'] = (0.9, 0.9)
        self.vertex_data[4]['in_color'] = (1.0, 1.0, 0.0)  # Yellow
        self.vertex_data[4]['in_size'] = PARTICLE_SIZES['DEBUG']
    
    def _key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        # Track key state
        if action == glfw.PRESS:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False
            
        # Handle one-time key presses
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_HOME:
                self.center_x = self.sim.box_size / 2
                self.center_y = self.sim.box_size / 2
                self.zoom = 10.0
            elif key == glfw.KEY_D:
                self.show_debug = not self.show_debug
            elif key == glfw.KEY_L:
                self.show_labels = not self.show_labels
            elif key == glfw.KEY_TAB:
                self.show_legend = not self.show_legend
    
    def _resize_callback(self, window, width, height):
        """Handle window resize."""
        self.width = max(1, width)
        self.height = max(1, height)
        self.ctx.viewport = (0, 0, width, height)
    
    def _process_input(self):
        """Process continuous input."""
        # Pan with arrow keys - speed depends on zoom
        pan_speed = self.sim.box_size / (10.0 * self.zoom)
        
        if self.keys.get(glfw.KEY_UP, False):
            self.center_y += pan_speed
        if self.keys.get(glfw.KEY_DOWN, False):
            self.center_y -= pan_speed
        if self.keys.get(glfw.KEY_LEFT, False):
            self.center_x -= pan_speed
        if self.keys.get(glfw.KEY_RIGHT, False):
            self.center_x += pan_speed
            
        # Zoom with +/- keys
        if self.keys.get(glfw.KEY_EQUAL, False) or self.keys.get(glfw.KEY_KP_ADD, False):
            self.zoom *= 1.05
        if self.keys.get(glfw.KEY_MINUS, False) or self.keys.get(glfw.KEY_KP_SUBTRACT, False):
            self.zoom /= 1.05
            
        # Clamp zoom
        self.zoom = max(1.0, min(100.0, self.zoom))
    
    def _sim_to_ndc(self, x, y):
        """Convert simulation coordinates to NDC."""
        # Calculate view dimensions in sim units
        aspect = self.width / self.height
        view_height = self.sim.box_size / self.zoom
        view_width = view_height * aspect
        
        # Convert to NDC
        ndc_x = 2.0 * (x - (self.center_x - view_width/2)) / view_width - 1.0
        ndc_y = 2.0 * (y - (self.center_y - view_height/2)) / view_height - 1.0
        
        return ndc_x, ndc_y
    
    def _draw_legend(self):
        """Draw color legend explaining particle types."""
        if not self.show_legend:
            return
            
        # Create a legend surface with Pygame
        legend_width = 250
        legend_height = 280
        legend_surface = pygame.Surface((legend_width, legend_height), pygame.SRCALPHA)
        
        # Fill with semi-transparent background
        legend_surface.fill((0, 0, 0, 180))
        
        # Draw title
        self.font.render_to(legend_surface, (10, 10), "Particle Legend", (255, 255, 255))
        
        # Draw color squares and labels for each particle type
        y_pos = 40
        for i, (key, (name, desc)) in enumerate(PARTICLE_TYPES.items()):
            # Get RGB color
            color = [int(c * 255) for c in COLORS[key]]
            
            # Draw color square
            pygame.draw.rect(legend_surface, color, (10, y_pos, 20, 20))
            
            # Draw name and description
            self.font.render_to(legend_surface, (40, y_pos), f"{name}: {desc}", (255, 255, 255))
            
            y_pos += 25
        
        # Convert pygame surface to OpenGL texture
        legend_data = pygame.image.tostring(legend_surface, 'RGBA', True)
        
        # Create or update the texture
        if not hasattr(self, 'legend_texture'):
            self.legend_texture = self.ctx.texture((legend_width, legend_height), 4, legend_data)
        else:
            self.legend_texture.write(legend_data)
        
        # Create simple program for displaying the legend if it doesn't exist
        if not hasattr(self, 'legend_program'):
            vertex_shader = """
            #version 330
            in vec2 in_position;
            in vec2 in_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
            """
            
            fragment_shader = """
            #version 330
            uniform sampler2D texture0;
            in vec2 v_texcoord;
            out vec4 fragColor;
            void main() {
                fragColor = texture(texture0, v_texcoord);
            }
            """
            
            self.legend_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            # Legend quad in top-right corner
            pos_x = 0.7  # position in NDC (0.7 to 1.0)
            pos_y = 0.5  # position in NDC (0.5 to 1.0)
            vertices = np.array([
                # positions    # texture coords
                pos_x, pos_y,      0.0, 0.0,
                pos_x, 1.0,        0.0, 1.0,
                1.0, pos_y,        1.0, 0.0,
                1.0, 1.0,          1.0, 1.0,
            ], dtype='f4')
            
            self.legend_vbo = self.ctx.buffer(vertices)
            self.legend_vao = self.ctx.vertex_array(
                self.legend_program, 
                [(self.legend_vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )
        
        # Render the legend
        self.legend_texture.use(0)
        self.legend_program['texture0'] = 0
        self.legend_vao.render(moderngl.TRIANGLE_STRIP)

    def update(self):
        """Update visualization with current simulation data."""
        if glfw.window_should_close(self.window):
            return
            
        # Process input
        self._process_input()
        
        # Clear the screen
        self.ctx.clear(0.0, 0.05, 0.1, 1.0)
        
        # Calculate particle count
        debug_count = 5 if self.show_debug else 0
        particle_count = debug_count
        
        # Add debug particles if enabled
        if self.show_debug:
            self._add_debug_particles()
        
        # Get simulation data
        quark_positions, quark_flavors, quark_colors = self.sim.get_quark_data()
        baryon_positions, baryon_types = self.sim.get_baryon_data()
        electron_positions, electron_velocities = self.sim.get_electron_data()
        
        # Get the quark array directly from the simulation to access bound state
        quark_array = self.sim.quark_array
        
        # Total number of particles to render
        total_particles = len(quark_positions) + len(baryon_positions) + len(electron_positions)
        
        # Process quark data
        for i, (pos, flavor, color) in enumerate(zip(quark_positions, quark_flavors, quark_colors)):
            idx = i + debug_count
            
            # Convert to NDC
            ndc_x, ndc_y = self._sim_to_ndc(pos[0], pos[1])
            
            # Store position
            self.vertex_data[idx]['in_pos'] = (ndc_x, ndc_y)
            
            # MUCH LARGER quark size for better visibility
            self.vertex_data[idx]['in_size'] = 30.0 if quark_array[i]['is_bound'] else 25.0  # Increased from default
            
            # Brighter, more distinct colors for quarks
            if flavor == QuarkFlavor.UP:
                if color == QuarkColor.RED:
                    self.vertex_data[idx]['in_color'] = (1.0, 0.2, 0.2)  # Bright Red
                elif color == QuarkColor.GREEN:
                    self.vertex_data[idx]['in_color'] = (0.2, 1.0, 0.2)  # Bright Green
                else:  # BLUE
                    self.vertex_data[idx]['in_color'] = (0.2, 0.2, 1.0)  # Bright Blue
            else:  # DOWN
                if color == QuarkColor.RED:
                    self.vertex_data[idx]['in_color'] = (1.0, 0.5, 0.5)  # Softer Red
                elif color == QuarkColor.GREEN:
                    self.vertex_data[idx]['in_color'] = (0.5, 1.0, 0.5)  # Softer Green
                else:  # BLUE
                    self.vertex_data[idx]['in_color'] = (0.5, 0.5, 1.0)  # Softer Blue
            
            particle_count += 1
        
        # Process baryon data
        for i, (pos, baryon_type) in enumerate(zip(baryon_positions, baryon_types)):
            idx = i + len(quark_positions) + debug_count
            
            # Convert to NDC
            ndc_x, ndc_y = self._sim_to_ndc(pos[0], pos[1])
            
            # Store position
            self.vertex_data[idx]['in_pos'] = (ndc_x, ndc_y)
            
            # Set color based on baryon type
            if baryon_type == BaryonType.PROTON:
                self.vertex_data[idx]['in_color'] = COLORS['PROTON']
            elif baryon_type == BaryonType.NEUTRON:
                self.vertex_data[idx]['in_color'] = COLORS['NEUTRON']
            elif baryon_type == BaryonType.DELTA_PLUS_PLUS:
                self.vertex_data[idx]['in_color'] = COLORS['DELTA_PP']
            elif baryon_type == BaryonType.DELTA_MINUS:
                self.vertex_data[idx]['in_color'] = COLORS['DELTA_M']
            
            # Set point size - SCALE WITH ZOOM
            self.vertex_data[idx]['in_size'] = PARTICLE_SIZES['BARYON'] * (self.zoom / 10.0)
            
            particle_count += 1
        
        # Add electrons after quarks and baryons
        electron_offset = len(quark_positions) + len(baryon_positions) + debug_count
        for i, (pos, vel) in enumerate(zip(electron_positions, electron_velocities)):
            idx = i + electron_offset
            
            # Convert to NDC
            ndc_x, ndc_y = self._sim_to_ndc(pos[0], pos[1])
            
            # Store position
            self.vertex_data[idx]['in_pos'] = (ndc_x, ndc_y)
            
            # Yellow color for electrons
            self.vertex_data[idx]['in_color'] = (1.0, 1.0, 0.0)  # Bright yellow
            
            # Smaller size for electrons
            self.vertex_data[idx]['in_size'] = 15.0  # Smaller than quarks
            
            particle_count += 1
        
        # Add visual indicators for baryon formation
        if not hasattr(self, 'formation_indicators'):
            self.formation_indicators = []
            self.last_baryon_count = 0
        
        # Check for newly formed baryons
        current_baryon_count = len(self.sim.baryons)
        if current_baryon_count > self.last_baryon_count:
            # Add a "Baryon Formed!" notification
            baryon_diff = current_baryon_count - self.last_baryon_count
            print(f"New baryons formed: {baryon_diff} (Total: {current_baryon_count})")
            self.formation_indicators.append({
                "text": f"+{baryon_diff} Baryons",
                "time": time.time(),
                "pos": (self.width/2, self.height*0.1)
            })
            self.last_baryon_count = current_baryon_count
        
        # Display indicators for a short time
        current_time = time.time()
        for indicator in list(self.formation_indicators):
            if current_time - indicator["time"] > 2.0:
                self.formation_indicators.remove(indicator)
        
        # Upload data and render
        self.vbo.write(self.vertex_data[:particle_count].tobytes())
        self.vao.render(moderngl.POINTS, vertices=particle_count)
        
        # Draw legend overlay after particles
        self._draw_legend()
        
        # Update FPS counter
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
        # Update window title with helpful info
        title = f"PyQMD - Quarks: {len(quark_positions)} - Baryons: {len(baryon_positions)} - FPS: {self.fps:.1f}"
        glfw.set_window_title(self.window, title)
        
        # Finish frame
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def should_close(self):
        """Check if the window should close."""
        return glfw.window_should_close(self.window)
    
    def cleanup(self):
        """Release resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
        if hasattr(self, 'program'):
            self.program.release()
        if hasattr(self, 'legend_texture'):
            self.legend_texture.release()
        if hasattr(self, 'legend_vbo'):
            self.legend_vbo.release()
        if hasattr(self, 'legend_vao'):
            self.legend_vao.release()
        if hasattr(self, 'legend_program'):
            self.legend_program.release()
        glfw.terminate()
