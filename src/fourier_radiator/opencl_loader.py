import numpy as np
import pyopencl as cl
from mako.template import Template

class OpenCLKernelLoader:
    def __init__(self, kernel_path, template_args, kernel_names=None, ctx=None):
        self.kernel_path = kernel_path
        self.template_args = template_args
        self.kernel_names = kernel_names  # optional expected kernel names
        self.ctx = ctx or cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        self._build_program()
        self._extract_kernels()

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
        "spectrum": cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=spectrum_flat),
        "x": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x),
        "y": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y),
        "z": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=z),
        "ux": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ux),
        "uy": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uy),
        "uz": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uz),
        "omega": cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omega),  # 转换为弧度
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
