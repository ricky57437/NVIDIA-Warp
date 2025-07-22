import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import math

from datetime import datetime

from pxr import Gf


class Example:
    def __init__(self, radius, stage_path="comp.usd"):
        #initial settings, setup such as the fps and radius of the particles.
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 128
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.total_step_time = 0.0

        self.radius = radius

        #initialize the model builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius
        builder.default_particle_color = Gf.Vec4f(0.0, 1.0, 0.0, 1.0)
        
        #make the particle grid, dimensions given and the cell sizes.
        builder.add_particle_grid(
            #dimensions of the rectangular grid
            dim_x = int(10 // (2 * self.radius)),
            dim_y = int(20 // (2 * self.radius)),
            dim_z = int(10 // (2 * self.radius)),
            #the sizes of each cell in the grid? Why do we do this for all 3D?
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            #where the center is?
            pos=wp.vec3(-5.0, 2.5, -5.0),
            #rotation
            rot=wp.quat_identity(),
            #initial velocity of the particles (theyre moving right in the simulation)
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=3.0,
            jitter=self.radius * 0.1,
        )  
        
        #base
        builder.add_shape_box(
            pos=wp.vec3(0,1,0),
            hx=20,
            hy=1,
            hz=20,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=1e6,
            kd=1e5
        )
        #top
        builder.add_shape_box(
            pos=wp.vec3(0,31,0),
            hx=20,
            hy=1,
            hz=20,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=1e6,
            kd=1e5
        )
        #right
        builder.add_shape_box(
            pos=wp.vec3(21,16,0),
            hx=1,
            hy=16,
            hz=22,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=1e6,
            kd=1e5
        )
        #left
        builder.add_shape_box(
            pos=wp.vec3(-21,16,0),
            hx=1,
            hy=16,
            hz=22,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=1e6,
            kd=1e5
        )
        #front
        builder.add_shape_box(
            pos=wp.vec3(0,16,21),
            hx=20,
            hy=16,
            hz=1,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=1e6,
            kd=1e5
        )
        #back
        builder.add_shape_box(
            pos=wp.vec3(0,16,-21),
            hx=20,
            hy=16,
            hz=1,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=1e6,
            kd=1e5
        )

        #finaliszes the builder ive been creating this whole time
        self.model = builder.finalize()
        #physics for the particles themselves when bumping into each other

        #creating 2 simulation states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        #creates the integrator, which solves how particles move and interact over time.
        self.integrator = wp.sim.SemiImplicitIntegrator()

        #sets up usd output rendering to visulize the simulation.
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        #tells to use CUDA graph for performance if available.
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

#Should be using this in all simulations?
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        timer = wp.ScopedTimer("step", active=True)
        with timer:
            self.model.particle_grid.build(self.state_0.particle_q, self.radius)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt
        self.total_step_time += timer.elapsed  # Accumulate step time

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            # Render particles
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="comp.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=600, help="Total number of frames.")
    #parser.add_argument("--radius", type=float, default=1.0, help="Radius of the particles.")

    args = parser.parse_known_args()[0]
    #[.5: 2,000, .232558: 18,963, .108696: 194,672, .050633: 2,000,000, 023585: 20,000,000]
    rad = [0.5, .232558]
    tt = []
    for r in rad:
        t1 = datetime.now()  # Use full datetime
        print(f"Running simulation with radius: {r}")
        example = Example(radius=r, stage_path=str(r)+".usd")

        for _ in range(args.num_frames):
            example.step()
            # example.render()

        # if example.renderer:
        #     example.renderer.save()

        t2 = datetime.now()  # Use full datetime
        delta = t2 - t1
        particle_count = int(10 // (2 * r)) * int(20 // (2 * r)) * int(10 // (2 * r))
        tt.append(f"{particle_count} particles: {delta.total_seconds():.6f} seconds")
    
    print()
    print(tt)
   
    # with wp.ScopedDevice(args.device):
    #     example = Example(radius=5.0, stage_path=args.stage_path)

    #     for _ in range(args.num_frames):
    #         example.step()
    #         example.render()

    #     if example.renderer:
    #         example.renderer.save()

    

    #     if example.renderer:
    #         example.renderer.save()
