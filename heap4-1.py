import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import math
import random
from datetime import datetime

from pxr import Gf


class Example:
    def __init__(self, stage_path="heap4-1.usd"):
        #initial settings, setup such as the fps and radius of the particles.
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 100000
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.4

        #initialize the model builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius
        
        #make the particle grid, dimensions given and the cell sizes.
        builder.add_particle_grid(
            #dimensions of the rectangular grid
            dim_x=35, #how many can fit in 29 #58
            dim_y=90, #14
            dim_z=35, #58
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            #where the center is?
            pos=wp.vec3(-14.0, 64.5, -14.0), 
            #rotation
            rot=wp.quat_identity(),
            #initial velocity of the particles (theyre moving right in the simulation)
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=20,
            jitter=self.radius * 0.0001,
        )

        for i in range(len(builder.particle_q)):
            builder.particle_radius[i] = random.choice([.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,
                                                        .1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,
                                                        .1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.4])
            #can change other things from builder (model.py lines 1183-1188)
            #MAYBE THIS HELPS WITH HOLLOW?
            if builder.particle_radius[i] == 0.6:
                builder.particle_mass[i] = 20

        self.ke = 1e6
        self.kd = 1e4

        #big slant
        builder.add_shape_box(
            pos=wp.vec3(0,59,0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.radians(15)),
            hx=15,
            hy=1.5,
            hz=15,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            mu=0.0,
            ke=self.ke,
            kd=self.kd,
        )
        #small slant
        builder.add_shape_box(
            pos=wp.vec3(-17,51.235,0),
            rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.radians(15)),
            hx=2,
            hy=1.5,
            hz=16,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            mu=0.0, #0.5 #i think this is the only one that has effect tbh
            ke=self.ke,
            kd=self.kd,
        )
        #right wall
        builder.add_shape_box(
            pos=wp.vec3(14.8,85.882,0),
            hx=0.5,
            hy=29.764,
            hz=15,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #front wall
        builder.add_shape_box(
            pos=wp.vec3(0,85.882,14.8),
            hx=15,
            hy=29.764,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=self.ke,
            kd=self.kd,
        )
        #back wall
        builder.add_shape_box(
            pos=wp.vec3(-2.5,85.882,-14.8),
            hx=17.5,
            hy=29.764,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #small wall
        builder.add_shape_box(
            pos=wp.vec3(-14.5,52.235,0),
            hx=0.5,
            hy=4.482,
            hz=15,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #left wall
        builder.add_shape_box(
            pos=wp.vec3(-19.5,81.7,0),
            hx=0.5,
            hy=33.946,
            hz=15,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #front small
        builder.add_shape_box(
            pos=wp.vec3(-17,84.882,14.8),
            hx=3,
            hy=30.764,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=self.ke,
            kd=self.kd,
        )

        #back container
        builder.add_shape_box(
            pos=wp.vec3(16,30.853,14.8),
            hx=35,
            hy=17.5,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #front container
        builder.add_shape_box(
            pos=wp.vec3(16,35.853,19.3),
            hx=35,
            hy=22.5,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
            ke=self.ke,
            kd=self.kd,
        )
        #left container
        builder.add_shape_box(
            pos=wp.vec3(-19.5,35.853,17.05),
            hx=0.5,
            hy=22.5,
            hz=2.75,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #right container
        builder.add_shape_box(
            pos=wp.vec3(51,30.853,17.05),
            hx=0.5,
            hy=17.5,
            hz=2.75,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            ke=self.ke,
            kd=self.kd,
        )
        #slant container
        builder.add_shape_box(
            pos=wp.vec3(16,23.853,17.05),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.radians(-18)),
            hx=37,
            hy=.5,
            hz=1.75,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
            mu=0.2,
            ke=800,
            kd=100,
        )

        #finaliszes the builder ive been creating this whole time
        self.model = builder.finalize()
        #physics for the particles themselves when bumping into each other
        self.model.particle_kf = 50.0
        #resistance to sliding btwn two particles
        
        self.model.particle_ke = 1e8
        self.model.particle_kd = 1e5
        self.model.particle_mu = 0.1
        #mu: particle friction coefficient GAME CHANGER
        
        
        self.model.particke_cohesion = 0.0
        #cohesion: attraction between molecules of the same substance
        self.model.particle_adhesion = 0.0
        #adhesion: attraction between molecules of different substances

        

        """
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0
        """

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
        default="heap4-1.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=3000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        t1 = datetime.now()  # Use full datetime
        for iframe in range(args.num_frames):
            example.step()
            example.render()
            print(iframe)

        if example.renderer:
            example.renderer.save()

    t2 = datetime.now()  # Use full datetime
    delta = t2 - t1
    print(f"sim started at {t1.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"sim ended at {t2.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"sim took {delta.total_seconds():.6f} seconds")
