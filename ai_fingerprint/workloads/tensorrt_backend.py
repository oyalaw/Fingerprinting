from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Workload


class TensorRTWorkload(Workload):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        artifact = config["ai"]["model_artifact"]

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runtime requested but TensorRT and PyCUDA are not installed"
            ) from exc

        self.trt = trt
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)

        with open(artifact, "rb") as handle:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())

        if self.engine is None:
            raise RuntimeError(f"Unable to deserialize TensorRT engine: {artifact}")

        self.context = self.engine.create_execution_context()

    def infer(self, array: np.ndarray) -> np.ndarray:
        trt = self.trt
        cuda = self.cuda
        engine = self.engine
        context = self.context

        if hasattr(engine, "num_io_tensors"):
            return self._infer_v3(array)

        raise RuntimeError(
            "This reference TensorRT adapter targets the TensorRT 10 tensor API. "
            "For older TensorRT versions, adapt the binding code for your JetPack release."
        )

    def _infer_v3(self, array: np.ndarray) -> np.ndarray:
        trt = self.trt
        cuda = self.cuda
        engine = self.engine
        context = self.context

        input_names = []
        output_names = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                "Reference TensorRT adapter currently supports one input and one output tensor"
            )

        input_name = input_names[0]
        output_name = output_names[0]

        array = np.ascontiguousarray(array)
        context.set_input_shape(input_name, array.shape)

        output_shape = tuple(context.get_tensor_shape(output_name))
        output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
        output = np.empty(output_shape, dtype=output_dtype)

        d_input = cuda.mem_alloc(array.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, array, stream)
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        return output
