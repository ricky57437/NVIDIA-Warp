import warp as wp
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="example_granular.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.8

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        builder.add_particle_grid(
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=self.radius * 50.0,
            cell_y=self.radius * 50.0,
            cell_z=self.radius * 50.0,
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

        builder.add_particle_grid(
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=self.radius * 50.0,
            cell_y=self.radius * 50.0,
            cell_z=self.radius * 50.0,
            pos=wp.vec3(10.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(-5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        ) 

        self.model = builder.finalize()

        #sets gravity to zero?
        self.model.gravity = wp.vec3(0.0, 0.0, 0.0)

        #Spring stiffness constant, how stiff the contact is (higher kf: particles resist overlapping more)
        self.model.particle_kf = 25.0

        #damping constant, how much energy dissapates (higher kd: more dissapated)
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="rickyexample3_onept.usd",
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
