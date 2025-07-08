import numpy as np
import pyopencl as cl
from mako.template import Template
try:
    from mpi4py import MPI
    mpi_installed = True
except ImportError:
    mpi_installed = False

class OpenCLKernelLoader:
    def __init__(self, kernel_path, template_args, kernel_names=None, ctx=None):
        if mpi_installed:
            comm = MPI.COMM_WORLD
            self.rank, self.size = comm.Get_rank(), comm.Get_size()
        else:
            self.rank, self.size = 0, 1

        self.kernel_path = kernel_path
        self.template_args = template_args
        self.kernel_names = kernel_names  # optional expected kernel names
        # self.ctx = ctx or cl.create_some_context()
        self.ctx = self._create_context(ctx)
        self.queue = cl.CommandQueue(self.ctx)

        self._build_program()
        self._extract_kernels()

    def _create_context(self, ctx):
        """
        ctx:
            1) 已经创建好的 cl.Context → 直接返回
            2) 字符串 'gpu' / 'cpu'    → 按类型挑设备
            3) None                    → 默认先找 GPU, 再退 CPU
        """
        # --- 1. 调用方直接给了 Context ---
        if isinstance(ctx, cl.Context):
            self.plat_name = "Manual"
            return ctx

        # --- 2. 决定想要的 device_type ---
        if isinstance(ctx, str):
            want = ctx.lower()
            if want == "gpu":
                dev_type = cl.device_type.GPU
            elif want == "cpu":
                dev_type = cl.device_type.CPU
            else:
                raise ValueError("ctx must be 'gpu', 'cpu', a cl.Context or None")
        else:                       # ctx is None
            dev_type = cl.device_type.GPU      # 默认先找 GPU
            fallback = cl.device_type.CPU      # 找不到就用 CPU

        try:
            # --- 3. 枚举平台，按 rank 取设备 ---
            for plat in cl.get_platforms():
                devs = plat.get_devices(device_type=dev_type)
                if devs:
                    device = devs[self.rank % len(devs)]
                    self.plat_name = plat.name
                    return cl.Context([device])

            # --- 4. 如果默认模式且 GPU 没找到，降级 CPU ---
            if ctx is None and fallback:
                for plat in cl.get_platforms():
                    devs = plat.get_devices(device_type=fallback)
                    if devs:
                        device = devs[self.rank % len(devs)]
                        self.plat_name = plat.name
                        return cl.Context([device])

            raise RuntimeError("No matching OpenCL device found")
        
        except Exception as e:
            print(f"[OpenCL] context creation failed: {e}")
            self.plat_name = "None"
            # return None
            raise RuntimeError("OpenCL plat_name None")
            
    def _build_program(self):
        tpl = Template(filename=self.kernel_path)
        src = tpl.render(**self.template_args)
        self.kernel_source = src  # in case you want to debug later

        try:
            self.program = cl.Program(self.ctx, src).build()
        except Exception as e:
            print("[ERROR] OpenCL kernel build failed!")
            for dev in self.ctx.devices:
                print(f"Build log for {dev.name}:")
                print(self.program.get_build_info(dev, cl.program_build_info.LOG))
            raise e

    def _extract_kernels(self):
        self.kernels = {}
        available = self.program.get_info(cl.program_info.KERNEL_NAMES).split(";")
        for name in available:
            self.kernels[name] = getattr(self.program, name)

        if self.kernel_names:
            for name in self.kernel_names:
                if name not in self.kernels:
                    raise ValueError(f"[ERROR] Kernel '{name}' not found in compiled program.")

    def get_kernel(self, name):
        return self.kernels.get(name)

    def get_queue(self):
        return self.queue

    def get_context(self):
        return self.ctx

    def get_source(self):
        return self.kernel_source


import numpy as np
import pyopencl as cl

def run_opencl_total_kernel(ctx, queue, kernel,
                            spectrum, x, y, z, ux, uy, uz,
                            wp, itStart, itEnd, nSteps,
                            omega, sinTheta, cosTheta, sinPhi, cosPhi,
                            dt, nSnaps, itSnaps):

    # 数据准备
    dtype = np.float32  # 或 np.float64
    spectrum_flat = spectrum.ravel()

    # 创建 buffers
    mf = cl.mem_flags
    bufs = {
        "spectrum": cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(spectrum_flat)),
        "x": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x)),
        "y": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(y)),
        "z": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(z)),
        "ux": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(ux)),
        "uy": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(uy)),
        "uz": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(uz)),
        "omega": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omega * 2 * np.pi),  # 转换为弧度
        "sinTheta": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sinTheta),
        "cosTheta": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cosTheta),
        "sinPhi": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sinPhi),
        "cosPhi": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cosPhi),
        "itSnaps": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=itSnaps),
    }


    # 设置参数
    kernel.set_args(
        bufs["spectrum"],
        bufs["x"], bufs["y"], bufs["z"],
        bufs["ux"], bufs["uy"], bufs["uz"],
        dtype(wp),
        np.uint32(itStart), np.uint32(itEnd), np.uint32(nSteps),
        bufs["omega"],
        bufs["sinTheta"], bufs["cosTheta"],
        bufs["sinPhi"], bufs["cosPhi"],
        np.uint32(len(omega)), np.uint32(len(sinTheta)), np.uint32(len(sinPhi)),
        dtype(dt),
        np.uint32(nSnaps),
        bufs["itSnaps"]
    )

    global_size = (len(omega) * len(sinTheta) * len(sinPhi),)
    local_size, total_size = compute_wgs(global_size[0])
    # 启动 kernel
    cl.enqueue_nd_range_kernel(queue, kernel, (total_size,), (local_size,))
    cl.enqueue_copy(queue, spectrum_flat, bufs["spectrum"])
    queue.finish()

    # 恢复形状
    return spectrum_flat.reshape(spectrum.shape)

def compute_wgs(total_elements):
    default_wgs = 256
    if total_elements <= default_wgs:
        return total_elements, total_elements
    total = int(np.ceil(total_elements / default_wgs)) * default_wgs
    return default_wgs, total
