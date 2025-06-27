import warp as wp
import warp.sim
import warp.sim.render
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np


class Example:
    def __init__(self, stage_path="example_granular.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

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
            cell_x=self.radius * 2.1,
            cell_y=self.radius * 2.1,
            cell_z=self.radius * 2.1,
            #cell is how far apart the radii of each particle is
            #since the radii is 3, the centers of the particles are 6 units away from each other.
            #cells should be > 2
            pos=wp.vec3(0.0, 3.0, 0.0),
            #position of grid
            rot=wp.quat_identity(),
            #rotation around axes (x, y, z, w)
            #determines rotation: Sin(value/2)
            #quat_identity() = (0,0,0,1)  -  no rotation
            #quat(0, 0.5, 0, 1)  -  rotation of 45 degrees along y axis
            vel=wp.vec3(5.0, 0.0, 0.0),
            #initial velocity of particles
            #3 components/directions
            mass=0.1,
            #mass of particles
            jitter=self.radius * 0.1,
            #random displacement of particles to make more "realistic"
            #jitter implemented lines 4190-4203 model.py
            #moves pos by random number [0,1] * jitter
        )

        builder.add_particle_grid(
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=self.radius * 50.0,
            cell_y=self.radius * 50.0,
            cell_z=self.radius * 50.0,
            pos=wp.vec3(10.0, 3.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(-5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        ) 

        self.model = builder.finalize()
        #indicates that the model was finalized

        self.model.gravity = wp.vec3(0.0, 0.0, 0.0)
        #sets gravity to zero
        #default = (0.0, -9.80665, 0.0)

        self.model.particle_ke = 10
        #particle normal contact stiffness
        #ke is multiplied with the spring displacement to calculate the normal force of two colliding particles
        #implemented 48-74 integrator_euler.py
        
        self.model.particle_kd = 0
        #particle normal contact damping
        #kd is multiplied with relative velocity to calculate normal force as well

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
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
            #swap states
            #1 becomes the current and 0 becomes the new. keeps repeating until loop ends

    def step(self):
        with wp.ScopedTimer("step"):
        #times how long it takes to run this block of code
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            #builds cell grid to analyze what particles are neighboring each other (eularian approacj)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt


    def render(self):
    #defined in render_usd.py
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
        #times like before to see performance
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()
            #begins a new frame, renders it, then ends


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ricky_twoptcollision.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=400, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()

    # Print or export the position data
    #import csv

    #with open("particle_positions.csv", "w", newline="") as f:
        #writer = csv.writer(f)
        #writer.writerow(["time", "x1", "y1", "z1", "x2", "y2", "z2"])
        #writer.writerows(example.positions)

    #print("Saved particle positions to particle_positions.csv")
