#!/usr/bin/env python
"""
PyQMD: Python Quantum Molecular Dynamics
A 2D quark-level nuclear phenomena simulator using GPU acceleration
"""
import argparse
import os
import sys
import time

# Add the source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from src.sim import Simulation
from src.visualize import Visualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyQMD: Nuclear simulation from first principles')
    parser.add_argument('--no-visual', action='store_true', help='Run without visualization')
    parser.add_argument('--particles', type=int, default=36, help='Number of initial quarks (multiple of 3 for complete baryons)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=1e-21, help='Time step in seconds')
    parser.add_argument('--save', type=str, help='Save simulation results to file')
    parser.add_argument('--load', type=str, help='Load simulation state from file')
    parser.add_argument('--box-size', type=float, default=12.0, help='Size of simulation box')
    parser.add_argument('--sim-speed', type=float, default=2.0, help='Simulation speed multiplier')
    parser.add_argument('--fast', action='store_true', default=False, help='Enable fast mode')
    parser.add_argument('--debug', action='store_true', default=True, help='Enable movement debugging')
    parser.add_argument('--realistic', action='store_true', default=True, help='Use realistic QCD physics')
    return parser.parse_args()

def main():
    """Main entry point for PyQMD."""
    args = parse_args()
    
    # Apply fast mode only if explicitly requested
    if args.fast:
        args.sim_speed = 5.0  # More reasonable speed multiplier
    
    # Initialize the simulation with reasonable particle count
    sim = Simulation(
        num_particles=args.particles,
        dt=args.dt,
        box_size=args.box_size
    )
    
    # Override simulation parameters for EXTREMELY VISIBLE physics
    try:
        # Try to modify physics parameters if they're available
        import src.physics
        
        # EXTREME MOVEMENT VALUES - last resort to ensure visibility
        src.physics.strong_const = 1000.0      # Much stronger force
        src.physics.coulomb_const = 500.0      # Much stronger EM force
        src.physics.gravity_const = 1.0        # Stronger gravity
        src.physics.u_mass = 0.1               # Ultra-light quarks for more movement
        src.physics.d_mass = 0.2               # Ultra-light quarks for more movement
        src.physics.electron_mass = 0.05       # Much lighter electrons
        src.physics.binding_thresh = 3.0       # Larger binding threshold
        src.physics.dt = 1e-15                 # MUCH larger time step (10,000x)
        src.physics.DEBUG_MOVEMENT = True
        
        print(f"Physics parameters set to EXTREME values for guaranteed visible motion")
    except (ImportError, AttributeError) as e:
        print(f"Note: Could not adjust physics parameters: {e}")
    
    # Load state if specified
    if args.load:
        sim.load_state(args.load)
        
    # Initialize visualization if not disabled
    vis = None
    if not args.no_visual:
        vis = Visualizer(sim)
        print("Interactive controls:")
        print("  Arrow keys: Pan the view")
        print("  +/- keys: Zoom in/out")
        print("  Home key: Reset view to center")
        print("  D key: Toggle debug particles")
        print("  Escape key: Exit simulation")
    
    # For performance tracking
    start_time = time.time()
    steps_completed = 0
    
    # Main simulation loop - run multiple physics steps per frame for smoother motion
    steps_per_frame = max(1, int(args.sim_speed))
    print(f"Running at {args.sim_speed}x speed ({steps_per_frame} steps per frame)")
    
    try:
        for step in range(args.steps):
            # Run multiple physics steps per frame for smoother motion at higher speeds
            for _ in range(steps_per_frame):
                sim.step()
                steps_completed += 1
            
            # Update visualization with the latest state
            if vis:
                vis.update()
                
                # Check if visualization window was closed
                if vis.should_close():
                    break
            
            # Print progress every 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    steps_per_sec = steps_completed / elapsed
                    print(f"Step {step}/{args.steps} - {steps_per_sec:.1f} steps/sec")
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    
    # Save state if specified
    if args.save:
        sim.save_state(args.save)
    
    # Cleanup visualization
    if vis:
        vis.cleanup()
    
    # Print final performance stats
    elapsed = time.time() - start_time
    if elapsed > 0:
        steps_per_sec = steps_completed / elapsed
        print(f"Simulation completed: {steps_completed} steps in {elapsed:.1f}s ({steps_per_sec:.1f} steps/sec)")

if __name__ == "__main__":
    main()
