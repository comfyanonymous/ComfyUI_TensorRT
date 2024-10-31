import torch
import tensorrt as trt
import os
from typing import Optional
from tqdm import tqdm
from math import prod
import comfy.model_management
from .models import get_model_from_version, TRTModelUtil

trt.init_libnvinfer_plugins(None, "")
logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)


def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16


class TRTModel(torch.nn.Module):
    def __init__(self, model_helper: TRTModelUtil, *args, **kwargs) -> None:
        super(TRTModel, self).__init__()

        self.context = None
        self.engine = None

        self.model = model_helper
        self.dtype = self.model.get_dtype()
        self.device = comfy.model_management.get_torch_device()

        self.input_names = self.model.get_input_names()
        self.output_names = self.model.get_output_names()

        self.current_shape: tuple[int] = (0,)
        self.output_shapes: dict[str, tuple[int]] = {}
        self.curr_split_batch: int = 0

        self.zero_pool = None
        self.extra_inputs: dict[str, torch.Tensor] = {}

    # Sets up the builder to use the timing cache file, and creates it if it does not already exist
    def _setup_timing_cache(self, config: trt.IBuilderConfig, timing_cache_path: str):
        buffer = b""
        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from timing cache.".format(len(buffer)))
        else:
            print("No timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig, timing_cache_path: str):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _create_profile(self, builder, min_config, opt_config, max_config):
        profile = builder.create_optimization_profile()

        min_config = opt_config if min_config is None else min_config
        max_config = opt_config if min_config is None else max_config

        min_shapes = self.model.get_input_shapes(**min_config)
        opt_shapes = self.model.get_input_shapes(**opt_config)
        max_shapes = self.model.get_input_shapes(**max_config)
        for input_name in opt_shapes.keys():
            profile.set_shape(
                input_name,
                min_shapes[input_name],
                opt_shapes[input_name],
                max_shapes[input_name],
            )

        return profile

    def build(
        self,
        onnx_path: str,
        engine_path: str,
        timing_cache_path: str,
        opt_config: dict,
        min_config: Optional[dict] = None,
        max_config: Optional[dict] = None,
    ) -> bool:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # TRT conversion starts here
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(onnx_path)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print("ONNX load ERROR")
            return False

        config = builder.create_builder_config()
        self._setup_timing_cache(config, timing_cache_path)
        config.progress_monitor = TQDMProgressMonitor()
        profile = self._create_profile(builder, min_config, opt_config, max_config)
        config.add_optimization_profile(profile)
        config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)  # STRIP_PLAN

        self.engine = builder.build_serialized_network(network, config)
        if self.engine is None:
            raise Exception("Failed to build Engine")

        model = {
            "engine": torch.ByteTensor(
                bytearray(self.engine)
            ),  # TODO this isn't very efficient
            "config": self.model.to_dict(),
        }
        torch.save(model, engine_path)

        return True

    @torch.cuda.nvtx.range("set_bindings_shape")
    def set_bindings_shape(self, inputs):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // self.curr_split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    @torch.cuda.nvtx.range("setup_tensors")
    def setup_tensors(self, model_inputs):
        raise NotImplementedError

    @classmethod
    def load_trt_model(cls, engine_path, model_type):
        try:
            engine = torch.load(engine_path)
            config = engine["config"]
            model, helper = get_model_from_version(model_type, config)
            unet = cls(helper)
            unet.engine = runtime.deserialize_cuda_engine(
                engine["engine"].numpy().tobytes()
            )
        except:
            model, helper = get_model_from_version(model_type, {})
            unet = cls(helper)
            with open(engine_path, "rb") as f:
                unet.engine = runtime.deserialize_cuda_engine(f.read())

        if unet.engine is None:
            raise Exception("Failed to load Engine")

        unet.context = unet.engine.create_execution_context()
        model.diffusion_model = unet
        model.memory_required = (
            lambda *args, **kwargs: 0
        )  # always pass inputs batched up as much as possible
        return model

    @torch.cuda.nvtx.range("__call__")
    def __call__(self):
        raise NotImplementedError


class TRTDiffusionBackbone(TRTModel):

    @torch.cuda.nvtx.range("setup_tensors")
    def setup_tensors(self, model_inputs):
        self.current_shape = model_inputs["x"].shape
        self.extra_inputs = {}

        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        batch_size, _, height, width = self.current_shape
        height *= 8
        width *= 8
        _, context_len, _ = model_inputs["context"].shape
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]
        # Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                self.curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs)

        # Inputs missing, use zero
        max_memory = 0
        for name in self.input_names:
            if name in model_inputs:
                continue
            shape = self.model.get_input_shapes_by_key(
                name,
                batch_size=batch_size,
                height=height,
                width=width,
                context_len=context_len,
            )
            shape = (shape[0] // self.curr_split_batch, *shape[1:])
            self.context.set_input_shape(name, shape)
            self.extra_inputs[name] = 0
            max_memory = max(prod(shape), max_memory)
        self.zero_pool = torch.zeros(
            max_memory, device=self.device, dtype=self.dtype
        ).contiguous()

        self.output_shapes = {}
        for name in self.output_names:
            shape = list(self.engine.get_tensor_shape("h"))
            for idx in range(len(shape)):
                if shape[idx] == -1:
                    shape[idx] = model_inputs["x"].shape[idx]
                if idx == 0:
                    shape[idx] = batch_size
            self.output_shapes[name] = shape

    @torch.cuda.nvtx.range("__call__")
    def __call__(
        self,
        x,
        timesteps,
        context,
        y=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        if self.model.use_control and control is not None:
            for control_layer, control_tensors in control.items():
                for i, tensor in enumerate(control_tensors):
                    model_inputs[f"{control_layer}_control_{i}"] = tensor

        for k, v in kwargs.items():
            # TODO actually needed? model_inputs[k] = v
            pass

        if self.current_shape != x.shape:
            self.setup_tensors(model_inputs)

        model_inputs["h"] = torch.empty(
            self.output_shapes["h"],
            device=self.device,
            dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype("h")),
        ).contiguous()

        for k in model_inputs:
            trt_dtype = trt_datatype_to_torch(self.engine.get_tensor_dtype(k))
            if model_inputs[k].dtype != trt_dtype:
                model_inputs[k] = model_inputs[k].to(trt_dtype)

        torch.cuda.nvtx.range_push("infer")
        stream = torch.cuda.default_stream(x.device)
        for i in range(self.curr_split_batch):
            for k, v in model_inputs.items():
                self.context.set_tensor_address(
                    k, v[(v.shape[0] // self.curr_split_batch) * i :].data_ptr()
                )
            for k in self.extra_inputs.keys():
                self.context.set_tensor_address(k, self.zero_pool.data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.nvtx.range_pop()

        return model_inputs["h"]


class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False
