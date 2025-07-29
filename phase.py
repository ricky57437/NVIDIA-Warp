import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import random
import math

from pxr import Gf

#changing: dt, ke/kd, radius

class Example:
    def __init__(self, myke, mykd, pke, pkd, stage_path="phase.usd"):
        #initial settings, setup such as the fps and radius of the particles.
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 128
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        
        self.radius = 0.1

        #initialize the model builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius
        builder.default_particle_color = Gf.Vec4f(0.0, 1.0, 0.0, 1.0)
        

        #make the particle grid, dimensions given and the cell sizes.
        builder.add_particle_grid(
            #dimensions of the rectangular grid
            dim_x=1, 
            dim_y=50,
            dim_z=1,
            cell_x=.25 * 2.0,
            cell_y=.25 * 2.0,
            cell_z=.25 * 2.0,
            #where the center is?
            pos=wp.vec3(0, 22, 0), 
            #rotation
            rot=wp.quat_identity(),
            #initial velocity of the particles
            vel=wp.vec3(0.0, -5.0, 0.0),
            mass=20,
            jitter=self.radius * 0.0,
        )
        
        #big slant
        builder.add_shape_box(
            pos=wp.vec3(0,19.5,0),
            hx=5,
            hy=0.5,
            hz=5,
            # rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.radians(15)),
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            mu=0.0,
            ke=myke,
            kd=mykd
        )

        #finaliszes the builder ive been creating this whole time
        self.model = builder.finalize()
        #physics for the particles themselves when bumping into each other
        self.model.particle_kf = 50.0
        #resistance to sliding btwn two particles
        
        self.model.particle_ke = pke
        self.model.particle_kd = pkd

        
        
        self.model.particke_cohesion = 0.0
        #cohesion: attraction between molecules of the same substance
        self.model.particle_adhesion = 0.0
        #adhesion: attraction between molecules of different substances

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

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step"):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

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
        default="phase.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=250, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(myke=1e3, mykd=1e3, pke=1e6, pkd=1e5, stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
