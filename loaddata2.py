import os
import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.render

@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh: wp.uint64,
    margin: float, # how far particles can penetrate the mesh
    dt: float, # timestep
    # parameters,
):
    tid = wp.tid()
    # Gets the particle position and velocity

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, -9.8, 0.0) * dt - v * 0.1 * dt
    # gravity + damping
    xpred = x + v * dt
    # predicts the next position using euler integration

    max_dist = 1.5
    query = wp.mesh_query_point_sign_normal(mesh, xpred, max_dist)
    # checks if the predicted position is intersecting the mesh
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        # evaluates the mesh position AT the intersection point

        delta = xpred - p
        # distance from the predicted position to the mesh position
        dist = wp.length(delta) * query.sign
        # negative = inside the mesh, positive = outside
        err = dist - margin
        # how much the particle is inside the mesh

        if err < 0.0:
            n = wp.normalize(delta) * query.sign
            xpred = xpred - n * err
            # calculates normal force like spring?
            #doesnt look like this uses ke or kd at all. but maybe i could add it?

    v = (xpred - x) / dt
    x = xpred
    # updates position and velocity

    positions[tid] = x
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="loaddata2.usd"):
        # self.num_particles = 100
        self.sim_dt = 1.0 / 300.0
        # time step (1/60 = 60 FPS)
        self.sim_time = 0.0
        # ?
        self.sim_timers = {}
        # ?
        self.sim_margin = .5
        # how close particles can get to the mesh

        #physical objects
        usd_path = r"C:/Users/ricky/OneDrive/Desktop/pull from github/warp/warp/examples/assets/bowl2.usdc"

        usd_stage = Usd.Stage.Open(usd_path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/Sphere/Bowl"))  # Adjust based on usdview output
        usd_scale = 10.0

        bowl_points = usd_geom.GetPointsAttr().Get() * usd_scale

        # Move bowl up by 5 units (along y-axis)
        bowl_offset = wp.vec3(0.0, 15.0, 0.0)
        bowl_points = np.array(bowl_points) + np.array([bowl_offset[0], bowl_offset[1], bowl_offset[2]])

        self.mesh = wp.Mesh(
            points=wp.array(bowl_points, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        #loads mesh from USD file and creates an interacting object using triangle indices and points

        #particles
        x_dim = 1
        y_dim = 1
        z_dim = 1
        spacing = 1.0  # distance between particles
        offset = wp.vec3(-5.0, 11.0, 5.0)  # center offset

        grid_pos = []
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    pos = wp.vec3(i * spacing, j * spacing, k * spacing) + offset
                    grid_pos.append(pos)

        init_pos = np.array([[p[0], p[1], p[2]] for p in grid_pos], dtype=np.float32)
        # converts the list of positions to a numpy array
        init_vel = np.zeros_like(init_pos)
        init_vel[:, 2] = -2.0  # Z-axis velocity set to -5 for all particles
        self.num_particles = len(init_pos)

        self.positions = wp.from_numpy(init_pos, dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel, dtype=wp.vec3)
        
        # converts the numpy array into gpu array format

        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
            self.renderer.render_ground()

    def step(self):
        with wp.ScopedTimer("step", dict=self.sim_timers):
            self.mesh.refit()

            wp.launch(
                kernel=simulate,
                dim=self.num_particles,
                inputs=[self.positions, self.velocities, self.mesh.id, self.sim_margin, self.sim_dt],
            )

            self.sim_time += self.sim_dt

    def render(self):
        if not self.renderer:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(3.0, 0.4, 0.8),
            )
            self.renderer.render_points(
                name="particles",
                points=self.positions.numpy(),
                radius=self.sim_margin,
                colors=(0.4, 0.0, 0.0),
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stage_path", type=lambda x: None if x == "None" else str(x), default="loaddata2.usd")
    parser.add_argument("--num_frames", type=int, default=1000)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
