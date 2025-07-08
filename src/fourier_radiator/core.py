
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
        self.dtype = np.float32 if dtype == "float" else np.double
        self.kernel = loader.get_kernel("total")
        self.queue = loader.get_queue()
        self.ctx = loader.get_context()

    def compute_radiation(self, **kwargs):
        """
        Compute the radiation spectrum using keyword arguments.

        :param kwargs: A dictionary of parameters, including:
            - tracks (list): Particle track data, where each element of the list should be a tuple containing:
                - x (np.ndarray): x positions (array of shape (n_steps,))
                - y (np.ndarray): y positions (array of shape (n_steps,))
                - z (np.ndarray): z positions (array of shape (n_steps,))
                - ux (np.ndarray): x-components of velocity (array of shape (n_steps,))
                - uy (np.ndarray): y-components of velocity (array of shape (n_steps,))
                - uz (np.ndarray): z-components of velocity (array of shape (n_steps,))
                - wp (float): Weight of the particle
                - id_start (int): Starting index of the particle's track

            - omega (np.ndarray): Angular frequency values (array of shape (n_omega,))
            - theta (np.ndarray): Polar angle values (array of shape (n_theta,))
            - phi (np.ndarray): Azimuthal angle values (array of shape (n_phi,))
            - dt (float): Time step for each iteration
            - nSteps (int): Total number of simulation steps
            - itSnaps (np.ndarray): Snapshot time steps (array of shape (n_snaps,))
            - nSnaps (int): Number of snapshots to calculate

        :return: Radiation spectrum (np.ndarray) of shape (n_snaps, n_theta, n_phi, n_omega)
        """
        # Extract parameters from kwargs
        tracks = kwargs.get("tracks")
        omega = kwargs.get("omega")
        theta = kwargs.get("theta")
        phi = kwargs.get("phi")
        dt = kwargs.get("dt")
        # nSteps = kwargs.get("nSteps")
        # Default for itSnaps: range from 0 to nSteps - 1 if not provided
        # itSnaps = kwargs.get("itSnaps", range(0, nSteps))
        # Default for nSnaps: length of itSnaps if not provided
        nSnaps = kwargs.get("nSnaps", 1)

        # Additional processing here, similar to your original code...
        sinTheta = np.sin(theta, dtype=self.dtype)
        cosTheta = np.cos(theta, dtype=self.dtype)
        sinPhi = np.sin(phi, dtype=self.dtype)
        cosPhi = np.cos(phi, dtype=self.dtype)
        c = 299792458.0  # Speed of light in m/s

        No = len(omega)
        Nt = len(theta)
        Np = len(phi)
        self.total_weight = self.dtype(0.0)

        # Initialize spectrum array
        spectrum = np.zeros((nSnaps, Nt, Np, No), dtype=self.dtype)

        # Compute radiation for each track
        for track in tqdm(tracks, desc="Calculating spectrum"):
            x, y, z, ux, uy, uz, wp, id_start = track
            
            it_start = np.uint32(0)
            it_range = (0, x.size)
            snap_iterations = np.ascontiguousarray(np.linspace( *(it_range+(nSnaps+1, )), dtype=np.uint32)[1:])
            
            self.total_weight += wp

            # Run the OpenCL kernel for this track
            spectrum = run_opencl_total_kernel(
                self.ctx, self.queue, self.kernel, spectrum, 
                x, y, z, ux, uy, uz,
                wp, it_start, np.uint32(it_range[-1]), np.uint32(x.size),
                omega, sinTheta, cosTheta, sinPhi, cosPhi,
                c * dt, np.uint32(nSnaps), snap_iterations
            )

        return norm_val(spectrum.swapaxes(-3, -1))
