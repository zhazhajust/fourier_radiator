"""Top-level package for fourier_radiator."""

__author__ = """Jie Cai"""
__email__ = 'jiecai@stu.pku.edu.cn'
__version__ = '0.1.0'

from .main import FourierRadiator
from .config import RadiationConfig
from .opencl_env import OpenCLEnvironment
from .compiler import KernelCompiler
from .particle import ParticleProcessor
from .data_manager import RadiationDataManager

__all__ = ["FourierRadiator", "RadiationConfig", "OpenCLEnvironment",
              "KernelCompiler", "ParticleProcessor", "RadiationDataManager"]
