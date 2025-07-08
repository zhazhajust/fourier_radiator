from mako.template import Template
import pyopencl as cl

class KernelCompiler:
    def __init__(self, mode, dtype_str, ctx, src_path):
        self.mode = mode
        self.dtype_str = dtype_str
        self.ctx = ctx
        self.src_path = src_path
        self.program = self._build_kernel()

    def _build_kernel(self):
        if self.ctx is None:
            return None

        kernel_file = "kernel_farfield.cl" if self.mode == 'far' else "kernel_nearfield.cl"
        kernel_path = self.src_path + kernel_file

        try:
            src = Template(filename=kernel_path).render(
                my_dtype=self.dtype_str,
                f_native=''  # 可扩展，比如使用 native_sqrt 等 OpenCL native 函数
            )
            return cl.Program(self.ctx, src).build()
        except Exception as e:
            print(f"[KernelCompiler] Kernel compilation failed: {e}")
            raise
