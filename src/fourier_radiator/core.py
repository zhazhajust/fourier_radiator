
import numpy as np
from tqdm import tqdm
from .opencl_loader import OpenCLKernelLoader, run_opencl_total_kernel
from .utils import norm_val

from fourier_radiator import __path__ as src_path
src_path = src_path[0] + '/kernels/'

class FourierRadiator:
    def __init__(self, kernel_path = src_path+"kernel_farfield.cl", dtype="float"):
        """
        Initializes the FourierRadiator class, loading the OpenCL kernel.
        :param kernel_path: OpenCL kernel file path
        :param dtype: Data type ("float" or "double")
        """
        loader = OpenCLKernelLoader(
            kernel_path=kernel_path,
            template_args={"my_dtype": dtype, "f_native": ""},
            kernel_names=["total"]
        )
        self.dtype = np.float32 if dtype == "float" else np.float64
        self.kernel = loader.get_kernel("total")
        self.queue = loader.get_queue()
        self.ctx = loader.get_context()

    def compute_radiation(self, tracks, omega, theta, phi, dt, nSteps, itSnaps, nSnaps):
        """
        Compute the radiation spectrum.
        :param tracks: Particle track data
        :param omega: Angular frequency
        :param theta: Angle theta
        :param phi: Angle phi
        :param dt: Time step
        :param nSteps: Total steps
        :param itSnaps: Snapshot time steps
        :param nSnaps: Number of snapshots
        :return: Radiation spectrum data
        """
        sinTheta = np.sin(theta, dtype=self.dtype)
        cosTheta = np.cos(theta, dtype=self.dtype)
        sinPhi = np.sin(phi, dtype=self.dtype)
        cosPhi = np.cos(phi, dtype=self.dtype)
        c = 299792458.0  # Speed of light in m/s
        
        No = len(omega)
        Nt = len(theta)
        Np = len(phi)
        self.total_weight = 0.0

        spectrum = np.zeros((nSnaps, Nt, Np, No), dtype=self.dtype)

        for track in tqdm(tracks, desc="Calculating spectrum"):
            x, y, z, ux, uy, uz, wp, id_start = track
            ptItStart = id_start
            ptItEnd = ptItStart + len(x) - 1
            self.total_weight += wp

            spectrum = run_opencl_total_kernel(
                self.ctx, self.queue, self.kernel,
                spectrum, x, y, z, ux, uy, uz,
                wp, ptItStart, ptItEnd, nSteps,
                omega / c, sinTheta, cosTheta, sinPhi, cosPhi,
                c * dt, nSnaps, itSnaps
            )
        
        return norm_val(spectrum.swapaxes(-3, -1))
