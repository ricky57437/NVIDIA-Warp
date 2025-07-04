import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import math

from pxr import Gf


class Example:
    def __init__(self, stage_path="example_granular_collision_sdf.usd"):
        #initial settings, setup such as the fps and radius of the particles.
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.3

        #initialize the model builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius
        
        #make the particle grid, dimensions given and the cell sizes.
        builder.add_particle_grid(
            #dimensions of the rectangular grid
            dim_x=20,
            dim_y=37,
            dim_z=12,
            #the sizes of each cell in the grid? Why do we do this for all 3D?
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            #where the center is?
            pos=wp.vec3(-6.0, 48.0, -2.0),
            #rotation
            rot=wp.quat_identity(),
            #initial velocity of the particles (theyre moving right in the simulation)
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

        #bottom square
        builder.add_shape_box(
            pos=wp.vec3(10,10,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 4),
            hx=10.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(-10,10,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / (-4)),
            hx=10.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(11,29,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi * (3/4)),
            hx=9.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(-11,29,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi * (-3/4)),
            hx=9.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )   
        #top square
        builder.add_shape_box(
            pos=wp.vec3(11,47,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 4),
            hx=9.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(-11,47,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / (-4)),
            hx=9.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(10,66,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi * (3/4)),
            hx=10.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        builder.add_shape_box(
            pos=wp.vec3(-10,66,0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi * (-3/4)),
            hx=10.0 * math.sqrt(2),
            hy=.1,
            hz=5.0,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=True,
        )
        #invisible walls
        builder.add_shape_box(
            pos=wp.vec3(0,38,-6),
            hx=20.0,
            hy=38,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
        )
        builder.add_shape_box(
            pos=wp.vec3(0,38,6),
            hx=20.0,
            hy=38,
            hz=0.5,
            density=1,
            is_solid=True,
            body=-1,
            is_visible=False,
        )
        

        #finaliszes the builder ive been creating this whole time
        self.model = builder.finalize()
        #physics for the particles themselves when bumping into each other
        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

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
        default="hourglass1.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
