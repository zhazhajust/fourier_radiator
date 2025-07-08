import numpy as np

class RadiationConfig:
    def __init__(self, Args):
        self.Args = Args.copy()
        self.Args.setdefault('mode', 'far')
        self.Args.setdefault('dtype', 'float')

        self.dtype = np.double if self.Args['dtype'] == 'double' else np.single
        self.Args.setdefault('Features', [])

        self._setup_grid()
        self._generate_angular_grid()

    def _setup_grid(self):
        self.Args['gridNodeNums'] = self.Args['grid'][-1]
        self.Args['numGridNodes'] = int(np.prod(self.Args['gridNodeNums']))

        No = self.Args['gridNodeNums'][0]
        omega_min, omega_max = self.Args['grid'][0]

        omega = np.r_[omega_min:omega_max:No*1j]
        for feature in self.Args['Features']:
            if feature == 'wavelengthGrid':
                self.Args['wavelengths'] = np.r_[1./omega_max:1./omega_min:No*1j]
                omega = 1. / self.Args['wavelengths']
            elif feature == 'logGrid':
                d_log_w = np.log(omega_max / omega_min) / (No - 1)
                omega = omega_min * np.exp(d_log_w * np.arange(No))

        self.Args['omega'] = omega.astype(self.dtype)
        self.Args['dw'] = np.abs(omega[1:] - omega[:-1]) if No > 1 else np.array([1.], dtype=self.dtype)

    def _generate_angular_grid(self):
        if self.Args['mode'] == 'far':
            Nt, Np = self.Args['gridNodeNums'][1:]
            theta_min, theta_max = self.Args['grid'][1]
            phi_min, phi_max = self.Args['grid'][2]

            theta = np.r_[theta_min:theta_max:Nt*1j]
            phi = phi_min + (phi_max - phi_min) / Np * np.arange(Np)

            self.Args['theta'] = theta.astype(self.dtype)
            self.Args['phi'] = phi.astype(self.dtype)
        else:
            Nr, Np = self.Args['gridNodeNums'][1:]
            r_min, r_max = self.Args['grid'][1]
            phi_min, phi_max = self.Args['grid'][2]

            radius = np.r_[r_min:r_max:Nr*1j]
            phi = phi_min + (phi_max - phi_min) / Np * np.arange(Np)

            self.Args['radius'] = radius.astype(self.dtype)
            self.Args['phi'] = phi.astype(self.dtype)

    def get_args(self):
        return self.Args

    def get_dtype(self):
        return self.dtype
