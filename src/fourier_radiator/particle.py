import numpy as np

class ParticleProcessor:
    def __init__(self, config, opencl_env, kernel_program):
        self.config = config
        self.env = opencl_env
        self.program = kernel_program
        self.dtype = config.get_dtype()
        self.Args = config.get_args()
        self.queue = self.env.get_queue()

    def track_to_device(self, particleTrack):
        if len(particleTrack) != 8:
            raise ValueError("Each particleTrack must have 8 elements")

        x, y, z, ux, uy, uz, wp, it_start = particleTrack

        return [
            self.env.to_device(x, self.dtype),
            self.env.to_device(y, self.dtype),
            self.env.to_device(z, self.dtype),
            self.env.to_device(ux, self.dtype),
            self.env.to_device(uy, self.dtype),
            self.env.to_device(uz, self.dtype),
            self.dtype(wp),
            np.uint32(it_start)
        ]

    def process_track(self, particleTrack, radiation_data,
                    snap_iterations, nSnaps, it_range=None):

        x, y, z, ux, uy, uz, wp, it_start = particleTrack

        # -------- snap_iterations 与 it_range --------
        if it_range is None:
            it_start = np.uint32(0)
            it_range = (0, x.size)
            snap_iterations = np.ascontiguousarray(
                np.linspace(*it_range, nSnaps + 1, dtype=np.uint32)[1:]
            )
            snap_iterations = self.env.to_device(snap_iterations, np.uint32)

        # -------- 线程配置 --------
        Nn = self.Args['numGridNodes']
        WGS, WGS_tot = self.env.compute_wgs(Nn)

        # -------- 1-6 轨迹数组 --------
        args_track = [coord.data for coord in (x, y, z, ux, uy, uz)]

        # -------- 7-10 标量 --------
        args_track += [
            self.dtype(wp),                    # wp
            np.uint32(it_start),               # itStart
            np.uint32(it_range[-1]),           # itEnd
            np.uint32(x.size)                  # nSteps
        ]

        # -------- 11-15 角/频率轴缓冲 --------
        args_axes = [
            radiation_data['omega'].data,
            radiation_data['sinTheta'].data,
            radiation_data['cosTheta'].data,
            radiation_data['sinPhi'].data,
            radiation_data['cosPhi'].data
        ]

        # -------- 16-18 网格尺寸 --------
        nOmega, nTheta, nPhi = self.Args['gridNodeNums']
        args_res = [np.uint32(nOmega), np.uint32(nTheta), np.uint32(nPhi)]

        # -------- 19-21 其他 --------
        args_aux = [
            self.dtype(self.Args['timeStep']),  # dt
            np.uint32(nSnaps),                  # nSnaps
            snap_iterations.data                # itSnaps
        ]

        # -------- 合并并调用 --------
        args = args_track + args_axes + args_res + args_aux

        self.program.total(
            self.queue,
            (WGS_tot,), (WGS,),
            radiation_data['radiation']['total'].data,
            *args
        )
