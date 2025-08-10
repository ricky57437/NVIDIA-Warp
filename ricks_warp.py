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
        # particle_a → Array of particle accelerations.
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
    # pos of this particle
    v = particle_v[i]
    # vel of this particle

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
    neighbors = wp.hash_grid_query(grid, x, radius * 2.5)
    # finds neighbors within a certain radius (2.5x particle radius here)

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

        delta = x - p
        # vector of shortest distance from the particle to the mesh surface
        dist = wp.length(delta) * query.sign
        err = dist - margin  # penetration depth if negative
    
        if err < 0.0:
            
            n = wp.normalize(delta) * query.sign
            # Surface normal (unit vector)
            c = -err
            # Penetration depth (positive scalar)       
            
            v_rel = v  # if mesh is moving, subtract its velocity
            # Relative velocity at contact point (mesh assumed static here)
            v_n = wp.dot(v_rel, n) * n
            # Normal velocity component
            v_t = v_rel - v_n
            # Tangential velocity component
            # this should be 0 because mesh is static, but just in case

            f_n = k_contact * c * n
            # Hooke's law normal force
            f_d = -k_damp * v_n
            # Normal damping force
            f_t = -k_friction * v_t
            # Tangential damping (viscous friction)
            
            f_t_len = wp.length(f_t)
            f_n_len = wp.length(f_n)
            if f_t_len > k_mu * f_n_len:
                if f_t_len > 1e-6:
                    f_t = f_t * ((k_mu * f_n_len) / f_t_len)
                else:
                    f_t = wp.vec3(0.0, 0.0, 0.0)
            # Apply Coulomb friction limit
            
            f_total = f_n + f_d + f_t
            # Total collision force
            
            f = f + f_total
            # Add to particles total force

    particle_f[i] = f
    # writes the total force to the particle_f array (for every particle)


@wp.kernel
def integrate(
# using initial pos and vel, forces are calculated, then applied to find/update new pos and vel
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float, # timestep
    inv_mass: float, #  Inverse mass (1 / mass) of each particle — allows faster multiplication instead of division???
):
    tid = wp.tid()

    v_new = v[tid] + (0.5) * a[tid] * dt
    # v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    # x_new = x[tid] + v_new * dt
    # this is where the new position and velocity are calculated using forces
    # apply_forces and contact_force is already called
    a_new = f[tid] * inv_mass + gravity

    v[tid] = v_new
    a[tid] = a_new
    # updates particle position and velocity using forces (wrote equations used in ipad)

@wp.kernel
def halfstep(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    dt: float, # timestep,
):
    tid = wp.tid()

    v_new = v[tid] + (0.5) * a[tid] * dt
    x_new = x[tid] + v_new * dt

    x[tid] = x_new
    # updates particle position and velocity to half steps

class Example:
    def __init__(self, stage_path="ricks_warp.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps # time step

        self.sim_substeps = 64 # number of calculations per frame
        self.sim_dt = self.frame_dt / self.sim_substeps # time step for each substep
        self.sim_time = 0.0

        self.point_radius = 0.5 # particle radius

        self.k_contact = 5000.0
        self.k_damp = 50.0
        self.k_friction = 0.5
        self.k_mu = 0.6  # for cohesive materials

        kc_p = 8000
        kd_p = 2
        kf_p = 1
        km_p = 1e5
        # particle coefficients

        kc_g = 8000
        kd_g = 2
        kf_g = 1
        km_g = 1e5
        # ground coefficients

        kc_m = 8000
        kd_m = 2
        kf_m = 1
        km_m = 1e5
        # mesh coefficients

        self.inv_mass = 64.0
        # not sure what this is for

        self.grid = wp.HashGrid(128, 128, 128) 
        # creates a hash grid for particle positions
        self.grid_cell_size = self.point_radius * 5.0

        self.points1 = self.particle_grid(10, 5, 30, (-30, 62, -13), self.point_radius, 0.1)
        # (x, y, z, corner, radius, jitter)
        # creates a particle grid
        # method is defined below

        self.x = wp.array(self.points1, dtype=wp.vec3)
        self.v = wp.array(np.ones([len(self.x), 3]) * np.array([0.0, 1.0, 0.0]), dtype=wp.vec3)
        self.a = wp.array(np.zeros([len(self.x), 3], dtype=np.float32) * np.array([0.0, 0.0, 0.0]),dtype=wp.vec3)
        self.a.numpy()[0] = [0.0, -9.8, 0.0]
        # initial velocity of particles
        # would be interesting on how to have two different initial velocities for different particles

        self.f = wp.zeros_like(self.v)

        #physical objects
        usd_path = r"C:/Users/ricky/OneDrive/Desktop/pull from github/warp/warp/examples/assets/plinko.usdc"

        usd_stage = Usd.Stage.Open(usd_path)
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/Plinko/Object"))  # Adjust based on usdview output
        usd_scale = 10.0

        mesh_points = usd_geom.GetPointsAttr().Get() * usd_scale

        mesh_offset = wp.vec3(0.0, 35.0, -30.0)
        # Move bowl up by 15 units (along y-axis)
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
                kernel=halfstep,
                dim=len(self.x),
                inputs=[self.x,
                        self.v,
                        self.a,
                        self.sim_dt
                ],
            )
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
                inputs=[self.x, self.v, self.a, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass],
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
                colors=(0.06, 0.65, 0.06)
            )

            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(0.74, 0.87, 0.41),
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
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=800, 
        # number of frames to simulate
        help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()

