import numpy as np

class RadiationDataManager:
    def __init__(self, config, opencl_env):
        self.config = config
        self.env = opencl_env
        self.Args = config.get_args()
        self.dtype = config.get_dtype()
        self.queue = self.env.get_queue()

        self.Data = {}
        self._init_grid_axes()
        self._init_radiation_buffer()

    def _init_grid_axes(self):
        if self.env.get_platform_name() == "None":
            return

        self.Data['omega'] = self.env.to_device(2 * np.pi * self.Args['omega'], self.dtype)

        if self.Args['mode'] == 'far':
            self.Data['sinTheta'] = self.env.to_device(np.sin(self.Args['theta']), self.dtype)
            self.Data['cosTheta'] = self.env.to_device(np.cos(self.Args['theta']), self.dtype)
            self.Data['sinPhi'] = self.env.to_device(np.sin(self.Args['phi']), self.dtype)
            self.Data['cosPhi'] = self.env.to_device(np.cos(self.Args['phi']), self.dtype)
        else:
            self.Data['radius'] = self.env.to_device(self.Args['radius'], self.dtype)
            self.Data['sinPhi'] = self.env.to_device(np.sin(self.Args['phi']), self.dtype)
            self.Data['cosPhi'] = self.env.to_device(np.cos(self.Args['phi']), self.dtype)

    def _init_radiation_buffer(self):
        self.Data['radiation'] = {}

    def prepare_radiation(self, sigma_particle, nSnaps):
        shape = (nSnaps,) + tuple(self.Args['gridNodeNums'][::-1])

        exp_factor = self.dtype(-0.5) * (2 * np.pi * self.Args['omega'] * sigma_particle) ** 2
        self.Data['FormFactor'] = self.env.to_device(np.exp(exp_factor), self.dtype)
        self.Data['radiation']['total'] = self.env.zeros(shape, dtype=self.dtype)

    def fetch_results(self):
        for key in self.Data['radiation']:
            arr = self.Data['radiation'][key].get()
            arr = arr.swapaxes(-1, -3)  # 调整轴顺序
            self.Data['radiation'][key] = np.ascontiguousarray(arr, dtype=np.double)

    def get_snap_iterations(self, it_range, nSnaps):
        snap_iterations = np.ascontiguousarray(
            np.linspace(*it_range, nSnaps + 1, dtype=np.uint32)[1:]
        )
        
        return self.env.to_device(snap_iterations, np.uint32)

    def get_data(self):
        return self.Data
