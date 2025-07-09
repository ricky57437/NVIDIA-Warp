import warp as wp
import warp.sim
import warp.sim.render
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

class Example:
    def __init__(self, stage_path="binary.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 128
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        #num frames: total number of frames (lenght of simulation)
        #sim_substeps: number of physics calculation substeps per frame (physical accuracy)
        #fps: frames per second (visual accuracy)

        self.radius = 3.0

        self.positions = []  # store (frame_number, pos1, pos2)

        builder = wp.sim.ModelBuilder()
        #instantiates ModelBuilder class to create a  binary collision scene
        builder.default_particle_radius = self.radius
        #changes default_particle_radius to 3.0
    
        
        builder.add_particle_grid(
        #method of ModelBuilder that takes in all the parameters below
            dim_x=1,
            dim_y=1,
            dim_z=1,
            #dim is how many particles along corresponding axis
            cell_x=1,
            cell_y=1,
            cell_z=1,
            #cell is how far apart the center of each particle is (must be > radius)
            pos=wp.vec3(-15.0, 3.0, 0.0),
            #position of grid
            rot=wp.quat_identity(),
            #rotation around axes (x, y, z, w)
            #determines rotation: Sin(value/2)
            #quat_identity() = (0,0,0,1)  -  no rotation
            #quat(0, 0.5, 0, 1)  -  rotation of 45 degrees along y axis
            vel=wp.vec3(8.0, 0.0, 0.0),
            #initial velocity of particles
            #3 components/directions
            mass=1,
            #mass of particles
            jitter=0,
            #random displacement of particles to make more "realistic"
            #jitter implemented lines 4190-4203 model.py
            #moves pos by random number [0,1] * jitter
        )


        builder.add_particle_grid(
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=1,
            cell_y=1,
            cell_z=1,
            pos=wp.vec3(15.0, 3.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(-8.0, 0.0, 0.0),
            mass=1,
            jitter=0,
        ) 

        self.model = builder.finalize()
        #indicates that the model was finalized

        self.model.gravity = wp.vec3(0.0, 0.0, 0.0)
        #sets gravity to zero
        #default = (0.0, -9.80665, 0.0)

        self.model.particle_ke = 100
        #particle normal contact stiffness
        #ke is multiplied with the spring displacement to calculate the normal force of two colliding particles
        #default value = 1.0e3
        #integrator_euler.py lines 48-74
    
        self.model.particle_kd = 0
        #particle normal contact damping
        #kd is multiplied with relative velocity to calculate normal force as well
        #default value = 1.0e2

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        #creates two simulation states

        self.integrator = wp.sim.SemiImplicitIntegrator()
        #instantiates SemiImplicitIntegrator
        #advances btwn the two states
        
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        #if a stage path is defined (this case it is to "example_granular.usd") then calls SimRenderer
        #parameters for SimRenderer(model.py, "example_granular.usd", scale 20:1 makes bigger to visualize easier)
        
        else:
            self.renderer = None
        #if no stage path defined, fails

        self.use_cuda_graph = wp.get_device().is_cuda
        #checks if cuda gpu is present
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
            #confused on what scopedcapture does
                self.simulate()
            self.graph = capture.graph
        #data from within that part of the simulation is stored into a cuda graph

    def simulate(self):
        for _ in range(self.sim_substeps):
        #sim_substeps = 64 (so repeats 64 times)
            self.state_0.clear_forces()
            #resets state_0, clean slate
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            #integrator.simulate(model, state in, state out, time step (sec))
            #using state in, calculates forces for state out
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
            #swap states
            #1 becomes the current and 0 becomes the new. keeps repeating until loop ends

    def step(self):
        with wp.ScopedTimer("step"):
        #times how long it takes to run this block of code
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            #builds cell grid to analyze what particles are neighboring each other (eularian approach)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            #runs the graph efficiently using cuda graph or by self.simulate which is constantly updating forces and pos

        self.sim_time += self.frame_dt


    def render(self):
    #defined in `render_usd.py
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
        #times like before to see performance
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()
            #by calling all these methods, begins a new frame, renders it, then ends


if __name__ == "__main__":
#only run this code if this file is being called directly (main function)
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #creates a new argument parser
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="binary.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=200, help="Total number of frames.")
    #tells the parser what device, stage, and how many frames to run on
    #num frames determines the length of the simulation. more frames = longer simulation

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
    #using the args default device, like gpu for example
        example = Example(stage_path=args.stage_path)
        #this runs the Example class above, setting up all the particles, usd stage, and environment

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
