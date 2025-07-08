import pyopencl as cl
import pyopencl.array as arrcl
import numpy as np

class OpenCLEnvironment:
    def __init__(self, rank, ctx=None):
        self.rank = rank
        self.ctx = self._create_context(ctx)
        self.queue = cl.CommandQueue(self.ctx)
        self.WGS = 256  # 默认工作组大小

    # def _create_context(self, ctx):
    #     if ctx is not None:
    #         self.plat_name = "Manual"
    #         return ctx

    #     try:
    #         platforms = cl.get_platforms()
    #         gpus = platforms[0].get_devices(device_type=cl.device_type.GPU)
    #         device = gpus[self.rank % len(gpus)]
    #         self.plat_name = device.platform.vendor
    #         return cl.Context(devices=[device])
    #     except Exception as e:
    #         print(f"OpenCL context creation failed: {e}")
    #         self.plat_name = "None"
    #         return None
        
    def _create_context(self, ctx):
        """
        ctx:
            1) 已经创建好的 cl.Context → 直接返回
            2) 字符串 'gpu' / 'cpu'    → 按类型挑设备
            3) None                    → 默认先找 GPU，再退 CPU
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
            return None

    def to_device(self, array, dtype=None):
        if dtype:
            array = np.ascontiguousarray(array.astype(dtype))
        return arrcl.to_device(self.queue, array)

    def zeros(self, shape, dtype):
        return arrcl.zeros(self.queue, shape, dtype=dtype)

    def get_queue(self):
        return self.queue

    def get_context(self):
        return self.ctx

    def get_platform_name(self):
        return self.plat_name

    def get_wgs(self):
        return self.WGS

    def compute_wgs(self, total_elements):
        if total_elements <= self.WGS:
            return total_elements, total_elements
        total = int(np.ceil(total_elements / self.WGS)) * self.WGS
        return self.WGS, total
