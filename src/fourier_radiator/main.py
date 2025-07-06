"""Main module."""

import numpy as np
from tqdm import tqdm

try:
    from mpi4py import MPI
    mpi_installed = True
except ImportError:
    mpi_installed = False

from .config import RadiationConfig
from .opencl_env import OpenCLEnvironment
from .compiler import KernelCompiler
from .particle import ParticleProcessor
from .data_manager import RadiationDataManager

src_path = "./kernels/"

class FourierRadiator:
    def __init__(self, Args):
        self.rank, self.size = self._get_mpi_info()
        
        self.config = RadiationConfig(Args)
        self.Args = self.config.get_args()
        self.dtype = self.config.get_dtype()

        self.env = OpenCLEnvironment(self.rank, self.Args.get("ctx"))
        self.compiler = KernelCompiler(self.Args['mode'], self.Args['dtype'], self.env.get_context(), src_path)
        self.processor = ParticleProcessor(self.config, self.env, self.compiler.program)
        self.data_mgr = RadiationDataManager(self.config, self.env)

    def calculate_spectrum(self, particleTracks, timeStep=None,
                           L_screen=None, Np_max=None, it_range=None,
                           nSnaps=1, sigma_particle=0,
                           weights_normalize=None,
                           verbose=True):

        if self.Args['mode'] == 'near':
            if L_screen is not None:
                self.Args['L_screen'] = L_screen
            else:
                raise ValueError("Define L_screen for near-field calculation")

        if timeStep is not None:
            self.Args['timeStep'] = self.dtype(timeStep)

        self.data_mgr.prepare_radiation(sigma_particle=self.dtype(sigma_particle), nSnaps=np.uint32(nSnaps))
        self.Data = self.data_mgr.get_data()

        if it_range is not None:
            self.snap_iterations = self.data_mgr.get_snap_iterations(it_range, nSnaps)
        else:
            self.snap_iterations = None
            if self.rank == 0 and verbose:
                print("Using individual it_range per track")

        # 选择粒子
        Np = len(particleTracks)
        if Np_max is not None:
            Np = min(Np_max, Np)
        particleTracks = particleTracks[:Np][self.rank::self.size]

        # 权重归一化
        self.total_weight = 0.0
        if weights_normalize in ['mean', 'max']:
            weights = [track[6] for track in particleTracks]
            norm = np.mean(weights) if weights_normalize == 'mean' else np.max(weights)
        else:
            norm = None

        iterator = tqdm(range(len(particleTracks))) if self.rank == 0 else range(len(particleTracks))
        for i in iterator:
            track = particleTracks[i]

            if weights_normalize == 'ones':
                track[6] = 1.0
            elif norm is not None:
                track[6] /= norm

            self.total_weight += track[6]

            device_track = self.processor.track_to_device(track)
            self.processor.process_track(device_track, self.Data, self.snap_iterations, nSnaps, it_range)

        self.data_mgr.fetch_results()

        if mpi_installed:
            self._gather_result_mpi()

    def _get_mpi_info(self):
        if mpi_installed:
            comm = MPI.COMM_WORLD
            return comm.Get_rank(), comm.Get_size()
        else:
            return 0, 1

    def _gather_result_mpi(self):
        comm = MPI.COMM_WORLD
        for key in self.Data['radiation']:
            buff = np.zeros_like(self.Data['radiation'][key])
            comm.Barrier()
            comm.Reduce([self.Data['radiation'][key], MPI.DOUBLE],
                        [buff, MPI.DOUBLE])
            comm.Barrier()
            self.Data['radiation'][key] = buff

        comm.Barrier()
        self.total_weight = comm.reduce(self.total_weight)
