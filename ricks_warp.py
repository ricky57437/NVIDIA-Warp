import os
import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.render


@wp.func
def contact_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
# computes the contact force between a particle and a surface/particle/(no mesh yet)
# normal component and friction component
        # n: Surface normal vector (unit vector pointing away from the surface).
        # v: Relative velocity at the contact point (particle velocity relative to surface).
        # c: Penetration depth (how far the particle is inside the surface).
        # k_n: Spring stiffness for normal force (like Hooke’s law).
        # k_d: Damping coefficient for normal direction.
        # k_f: Tangential damping (for friction).
        # k_mu: Friction coefficient (Coulomb friction).
    vn = wp.dot(n, v)
    # how far the particle is moving away or into the surface/particle (normal component of relative velocity)
    jn = c * k_n
    # spring force (normal component))
    jd = min(vn, 0.0) * k_d
    # damping force

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    # tangential component of the relative velocity
    vs = wp.length(vt)
    #magnitude of the tangential velocity?

    if vs > 0.0:
        vt = vt / vs
        #gives a unit vector in the direction of friction

    # Coulomb condition
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))

    # total force
    return -n * fn - vt * ft


@wp.kernel
def apply_forces(
# apply ground contact / particle to particle forces using the method above
    grid: wp.uint64,
    mesh: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    radius: float,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
):
        # grid → A Warp hash grid structure storing particle positions, used for fast neighbor lookups.
        # particle_x → Array of particle positions.
        # particle_v → Array of particle velocities.
        # particle_f → Array where we’ll write the total force for each particle.
        # radius → The particle radius (for detecting overlaps).
        # k_contact → Normal spring stiffness.
        # k_damp → Normal damping coefficient.
        # k_friction → Tangential damping (friction) coefficient.
        # k_mu → Friction coefficient for Coulomb friction.
    
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    v = particle_v[i]

    f = wp.vec3()
    # total force vector is 0 at the start

    # ground contact
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x)
    #particles distance to ground (y=0 plane)

    cohesion_ground = 0.0
    cohesion_particle = 0.00
    # how far particles can penetrate the ground/each other before cohesion force is applied

    if c < cohesion_ground:
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)
        # if the particle is below the ground (y=0), apply contact force upwards
        # calls contact force here
        # normal is always (0,1,0) for ground, varies for particles

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, radius * 5.0)
    # finds neighbors within a certain radius (5x particle radius here)

    for index in neighbors:
        if index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius * 2.0

            if err <= cohesion_particle:
            # if the particles are within the cohesion distance, apply contact force
                n = n / d
                # normal vector between the two particles
                vrel = v - particle_v[index]
                # relative velocity between the two particles

                f = f + contact_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)
                # calls contact force here
    
    margin = radius * 0.5
    max_dist = 1.5
    query = wp.mesh_query_point_sign_normal(mesh, x, max_dist)
    if query.result:
        # Get exact hit position on mesh
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

        # Vector from mesh surface to particle
        delta = x - p
        dist = wp.length(delta) * query.sign
        err = dist - margin  # penetration depth if negative
    
        if err < 0.0:
            # Surface normal (unit vector)
            n = wp.normalize(delta) * query.sign

            # Penetration depth (positive scalar)
            c = -err

            # Relative velocity at contact point (mesh assumed static here)
            v_rel = v  # if mesh is moving, subtract its velocity

            # Normal velocity component
            v_n = wp.dot(v_rel, n) * n

            # Tangential velocity component
            v_t = v_rel - v_n

            # Hooke's law normal force
            f_n = k_contact * c * n

            # Normal damping force
            f_d = -k_damp * v_n

            # Tangential damping (viscous friction)
            f_t = -k_friction * v_t

            # Apply Coulomb friction limit
            f_t_len = wp.length(f_t)
            f_n_len = wp.length(f_n)
            if f_t_len > k_mu * f_n_len:
                if f_t_len > 1e-6:
                    f_t = f_t * ((k_mu * f_n_len) / f_t_len)
                else:
                    f_t = wp.vec3(0.0, 0.0, 0.0)

            # Total collision force
            f_total = f_n + f_d + f_t

            # Add to particle force accumulator
            f = f + f_total

    particle_f[i] = f
    # writes the total force to the particle_f array (for every particle)


@wp.kernel
def integrate(
# using initial pos and vel, forces are calculated, then applied to find/update new pos and vel
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float, # timestep
    inv_mass: float, #  Inverse mass (1 / mass) of each particle — allows faster multiplication instead of division???
):
    tid = wp.tid()

    v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    x_new = x[tid] + v_new * dt

    v[tid] = v_new
    x[tid] = x_new
    # updates particle position and velocity using forces (wrote equations used in ipad)


class Example:
    def __init__(self, stage_path="ricks_warp.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps # time step

        self.sim_substeps = 64 # number of calculations per frame
        self.sim_dt = self.frame_dt / self.sim_substeps # time step for each substep
        self.sim_time = 0.0

        self.point_radius = 0.1 # particle radius

        self.k_contact = 8000.0
        self.k_damp = 2.0
        self.k_friction = 1.0
        self.k_mu = 100000.0  # for cohesive materials

        self.inv_mass = 64.0

        self.grid = wp.HashGrid(128, 128, 128) # creates a hash grid for particle positions
        self.grid_cell_size = self.point_radius * 5.0

        self.points = self.particle_grid(32, 64, 32, (0.0, 0.5, 0.0), self.point_radius, 0.1)

        self.x = wp.array(self.points, dtype=wp.vec3)
        self.v = wp.array(np.ones([len(self.x), 3]) * np.array([0.0, 0.0, -15.0]), dtype=wp.vec3)
        self.f = wp.zeros_like(self.v)

        #physical objects
        usd_path = r"C:/Users/ricky/OneDrive/Desktop/pull from github/warp/warp/examples/assets/bowl2.usdc"

        usd_stage = Usd.Stage.Open(usd_path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/Sphere/Bowl"))  # Adjust based on usdview output
        usd_scale = 10.0

        mesh_points = usd_geom.GetPointsAttr().Get() * usd_scale

        # Move bowl up by 5 units (along y-axis)
        mesh_offset = wp.vec3(0.0, 15.0, 0.0)
        mesh_points = np.array(mesh_points) + np.array([mesh_offset[0], mesh_offset[1], mesh_offset[2]])

        self.mesh = wp.Mesh(
            points=wp.array(mesh_points, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
            self.renderer.render_ground()
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
    # for every substep, apply forces, then integrate to find new position and velocity
        for _ in range(self.sim_substeps):
            wp.launch(
                kernel=apply_forces,
                dim=len(self.x),
                inputs=[
                    self.grid.id,
                    self.mesh.id,
                    self.x,
                    self.v,
                    self.f,
                    self.point_radius,
                    self.k_contact,
                    self.k_damp,
                    self.k_friction,
                    self.k_mu,
                ],
            )
            wp.launch(
                kernel=integrate,
                dim=len(self.x),
                inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass],
            )

    def step(self):
        with wp.ScopedTimer("step"):
            with wp.ScopedTimer("grid build", active=False):
                self.grid.build(self.x, self.grid_cell_size)
                # rebuilds the hash grid with the updated particle positions

            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
                # for every frame, we run step then render

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            
            self.renderer.render_points(
                points=self.x.numpy(), 
                radius=self.point_radius, 
                name="points", 
                colors=(0.8, 0.3, 0.2)
            )

            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(3.0, 0.4, 0.8),
            )

            self.renderer.end_frame()

    # creates a grid of particles
    def particle_grid(self, dim_x, dim_y, dim_z, lower, radius, jitter):
        # lower: Lower corner of the grid (x, y, z).
        rng = np.random.default_rng(42)
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
        points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
        points_t = points_t + rng.random(size=points_t.shape) * radius * jitter

        return points_t.reshape((-1, 3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ricks_warp.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=200, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()

