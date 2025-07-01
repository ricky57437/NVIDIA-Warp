import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

from pxr import Gf


class Example:
    def __init__(self, stage_path="example_granular_collision_sdf.usd"):
        #initial settings, setup such as the fps and radius of the particles.
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.1

        #initialize the model builder
        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        #make the particle grid, dimensions given and the cell sizes.
        builder.add_particle_grid(
            #dimensions of the rectangular grid
            dim_x=5,
            dim_y=5,
            dim_z=5,
            #the sizes of each cell in the grid? Why do we do this for all 3D?
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            #where the center is?
            pos=wp.vec3(-1.0, 12.5, -1.0),
            #rotation
            rot=wp.quat_identity(),
            #initial velocity of the particles (theyre moving right in the simulation)
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

    # constructs a 3D signed distance field (SDF) for a box so that particles in the simulation can collide with it realistically.
        #setting a rect with the bounds:
        mins = np.array([-5.0, -5.0, -5.0])
        #resolution
        voxel_size = 0.2
        maxs = np.array([5.0, 5.0, 5.0])
        #determines how many voxels are within the SDF region
        nums = np.ceil((maxs - mins) / (voxel_size)).astype(dtype=int)

        # Rect dimensions
        rectx_half_extent = 2.5
        recty_half_extent = 0.2
        center = np.array([0.0, 0.0, 0.0])

        rect_sdf_np = np.zeros(tuple(nums))
        for x in range(nums[0]):
            for y in range(nums[1]):
                for z in range(nums[2]):
                    pos = mins + voxel_size * np.array([x, y, z])
                    #rel = how far pt is from the center (0,0,0)
                    rel = pos - center
                    dx = abs(rel[0]) - rectx_half_extent
                    dy = abs(rel[1]) - recty_half_extent
                    dz = abs(rel[2]) - rectx_half_extent

                    # Compute distance outside the rect
                    outside = np.maximum([dx, dy, dz], 0.0)
                    inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)

                    rect_sdf_np[x, y, z] = np.linalg.norm(outside) + inside
                        
        #given all parameters of sdf, finally creates the rect
        rect_vdb = wp.Volume.load_from_numpy(rect_sdf_np, mins, voxel_size, rectx_half_extent + 3.0 * voxel_size)
        #rect turns into a sim object for the simulation
        rect_sdf = wp.sim.SDF(rect_vdb)


        self.rect_pos = wp.vec3(0.0, 10.0, 0.0)
        self.rect_scale = 1.0
        self.rectx_half_extent = rectx_half_extent
        self.recty_half_extent = recty_half_extent
        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rect_sdf,
            body=-1,
            pos=self.rect_pos,
            scale=wp.vec3(1.0, 1.0, 1.0),
        )

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rect_sdf,
            body=-1,
            pos=(2.5,12.5,0),
            scale=wp.vec3(1.0, 1.0, 1.0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2),
        )

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rect_sdf,
            body=-1,
            pos=(-2.5,12.5,0),
            scale=wp.vec3(1.0, 1.0, 1.0),
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2),
        )

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rect_sdf,
            body=-1,
            pos=(0,12.5,2.5),
            scale=wp.vec3(1.0, 1.0, 1.0),
            rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2),
        )

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rect_sdf,
            body=-1,
            pos=(0,12.5,-2.5),
            scale=wp.vec3(1.0, 1.0, 1.0),
            rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2),
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

            # Render 5 walls of the rect (sides and bottom)
            wall_thickness = 0.1
            extentx = self.rectx_half_extent
            extenty = self.recty_half_extent
            base = self.rect_pos

            # bottom
            self.renderer.render_box(
                name="bottom",
                #the "center" of the wall (bottom)
                pos=(0,9.8,0),
                #extents that push out to create the wall from one single point
                extents=wp.vec3(2.5, 0.2, 2.5),
                rot=wp.quat_identity(),
                color=Gf.Vec3f(0.0, 0.0, 1.0),
                # RGB Blue
            )
            
            # back wall (Z-)
            self.renderer.render_box(
                name="back",
                pos=(0,12.5,-2.6),
                extents=wp.vec3(2.5,2.5,0.1),
                rot=wp.quat_identity(),
                color=Gf.Vec3f(0.0, 0.0, 0.5),
            )

            # front wall (Z+)
            self.renderer.render_box(
                name="front",
                pos=(0,12.5,2.6),
                extents=wp.vec3(2.5,2.5,-0.1),
                rot=wp.quat_identity(),
                color=Gf.Vec3f(0.0, 0.0, 0.5),
            )

            # left wall (X-)
            self.renderer.render_box(
                name="left",
                pos=(-2.6,12.5,0),
                extents=wp.vec3(0.1,2.5,2.5),
                rot=wp.quat_identity(),
                color=Gf.Vec3f(0.0, 0.0, 0.5),
            )

            # right wall (X+)
            self.renderer.render_box(
                name="right",
                pos=(2.6,12.5,0),
                extents=wp.vec3(-0.1,2.5,2.5),
                rot=wp.quat_identity(),
                color=Gf.Vec3f(0.0, 0.0, 0.5),
            )

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
        default="rectangle2.usd",
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
