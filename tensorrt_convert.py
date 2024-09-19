import torch
import sys
import os
import time
import comfy.model_management

import tensorrt as trt
import folder_paths
from tqdm import tqdm

# TODO:
# Make it more generic: less model specific code

# add output directory to tensorrt search path
if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.get_output_directory(), "tensorrt")
    )
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.get_output_directory(), "tensorrt")],
        {".engine"},
    )

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
        

class TRT_MODEL_CONVERSION_BASE:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.timing_cache_path = os.path.normpath(
            os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timing_cache.trt"))
        )

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(s):
        raise NotImplementedError

    # Sets up the builder to use the timing cache file, and creates it if it does not already exist
    def _setup_timing_cache(self, config: trt.IBuilderConfig):
        buffer = b""
        if os.path.exists(self.timing_cache_path):
            with open(self.timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from timing cache.".format(len(buffer)))
        else:
            print("No timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(self.timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _convert(
        self,
        model,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        context_min,
        context_opt,
        context_max,
        num_video_frames,
        is_static: bool,
    ):
        output_onnx = os.path.normpath(
            os.path.join(
                os.path.join(self.temp_dir, "{}".format(time.time())), "model.onnx"
            )
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
        unet = model.model.diffusion_model

        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        context_len = 77
        context_len_min = context_len
        y_dim = model.model.adm_channels
        extra_input = {}
        dtype = torch.float16

        if isinstance(model.model, comfy.model_base.SD3): #SD3
            context_embedder_config = model.model.model_config.unet_config.get("context_embedder_config", None)
            if context_embedder_config is not None:
                context_dim = context_embedder_config.get("params", {}).get("in_features", None)
                context_len = 154 #NOTE: SD3 can have 77 or 154 depending on which text encoders are used, this is why context_len_min stays 77
        elif isinstance(model.model, comfy.model_base.AuraFlow):
            context_dim = 2048
            context_len_min = 256
            context_len = 256
        elif isinstance(model.model, comfy.model_base.Flux):
            context_dim = model.model.model_config.unet_config.get("context_in_dim", None)
            context_len_min = 256
            context_len = 256
            y_dim = model.model.model_config.unet_config.get("vec_in_dim", None)
            extra_input = {"guidance": ()}
            dtype = torch.bfloat16

        if context_dim is not None:
            input_names = ["x", "timesteps", "context"]
            output_names = ["h"]

            dynamic_axes = {
                "x": {0: "batch", 2: "height", 3: "width"},
                "timesteps": {0: "batch"},
                "context": {0: "batch", 1: "num_embeds"},
            }

            transformer_options = model.model_options['transformer_options'].copy()
            if model.model.model_config.unet_config.get(
                "use_temporal_resblock", False
            ):  # SVD
                batch_size_min = num_video_frames * batch_size_min
                batch_size_opt = num_video_frames * batch_size_opt
                batch_size_max = num_video_frames * batch_size_max

                class UNET(torch.nn.Module):
                    def forward(self, x, timesteps, context, y):
                        return self.unet(
                            x,
                            timesteps,
                            context,
                            y,
                            num_video_frames=self.num_video_frames,
                            transformer_options=self.transformer_options,
                        )

                svd_unet = UNET()
                svd_unet.num_video_frames = num_video_frames
                svd_unet.unet = unet
                svd_unet.transformer_options = transformer_options
                unet = svd_unet
                context_len_min = context_len = 1
            else:
                class UNET(torch.nn.Module):
                    def forward(self, x, timesteps, context, *args):
                        extras = input_names[3:]
                        extra_args = {}
                        for i in range(len(extras)):
                            extra_args[extras[i]] = args[i]
                        return self.unet(x, timesteps, context, transformer_options=self.transformer_options, **extra_args)

                _unet = UNET()
                _unet.unet = unet
                _unet.transformer_options = transformer_options
                unet = _unet

            input_channels = model.model.model_config.unet_config.get("in_channels", 4)

            inputs_shapes_min = (
                (batch_size_min, input_channels, height_min // 8, width_min // 8),
                (batch_size_min,),
                (batch_size_min, context_len_min * context_min, context_dim),
            )
            inputs_shapes_opt = (
                (batch_size_opt, input_channels, height_opt // 8, width_opt // 8),
                (batch_size_opt,),
                (batch_size_opt, context_len * context_opt, context_dim),
            )
            inputs_shapes_max = (
                (batch_size_max, input_channels, height_max // 8, width_max // 8),
                (batch_size_max,),
                (batch_size_max, context_len * context_max, context_dim),
            )

            if y_dim > 0:
                input_names.append("y")
                dynamic_axes["y"] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min, y_dim),)
                inputs_shapes_opt += ((batch_size_opt, y_dim),)
                inputs_shapes_max += ((batch_size_max, y_dim),)

            for k in extra_input:
                input_names.append(k)
                dynamic_axes[k] = {0: "batch"}
                inputs_shapes_min += ((batch_size_min,) + extra_input[k],)
                inputs_shapes_opt += ((batch_size_opt,) + extra_input[k],)
                inputs_shapes_max += ((batch_size_max,) + extra_input[k],)


            inputs = ()
            for shape in inputs_shapes_opt:
                inputs += (
                    torch.zeros(
                        shape,
                        device=comfy.model_management.get_torch_device(),
                        dtype=dtype,
                    ),
                )

        else:
            print("ERROR: model not supported.")
            return ()

        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)
        torch.onnx.export(
            unet,
            inputs,
            output_onnx,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        # TRT conversion starts here
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)

        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(output_onnx)
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

        if not success:
            print("ONNX load ERROR")
            return ()

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        self._setup_timing_cache(config)
        config.progress_monitor = TQDMProgressMonitor()

        prefix_encode = ""
        for k in range(len(input_names)):
            min_shape = inputs_shapes_min[k]
            opt_shape = inputs_shapes_opt[k]
            max_shape = inputs_shapes_max[k]
            profile.set_shape(input_names[k], min_shape, opt_shape, max_shape)

            # Encode shapes to filename
            encode = lambda a: ".".join(map(lambda x: str(x), a))
            prefix_encode += "{}#{}#{}#{};".format(
                input_names[k], encode(min_shape), encode(opt_shape), encode(max_shape)
            )

        if dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        if dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)

        config.add_optimization_profile(profile)

        if is_static:
            filename_prefix = "{}_${}".format(
                filename_prefix,
                "-".join(
                    (
                        "stat",
                        "b",
                        str(batch_size_opt),
                        "h",
                        str(height_opt),
                        "w",
                        str(width_opt),
                    )
                ),
            )
        else:
            filename_prefix = "{}_${}".format(
                filename_prefix,
                "-".join(
                    (
                        "dyn",
                        "b",
                        str(batch_size_min),
                        str(batch_size_max),
                        str(batch_size_opt),
                        "h",
                        str(height_min),
                        str(height_max),
                        str(height_opt),
                        "w",
                        str(width_min),
                        str(width_max),
                        str(width_opt),
                    )
                ),
            )

        serialized_engine = builder.build_serialized_network(network, config)

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        output_trt_engine = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.engine"
        )

        with open(output_trt_engine, "wb") as f:
            f.write(serialized_engine)

        self._save_timing_cache(config)

        return ()


class DYNAMIC_TRT_MODEL_CONVERSION(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(DYNAMIC_TRT_MODEL_CONVERSION, self).__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_DYN"}),
                "batch_size_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "batch_size_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height_max": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_min": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_max": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "context_min": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "context_max": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "num_video_frames": (
                    "INT",
                    {
                        "default": 14,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        model,
        filename_prefix,
        batch_size_min,
        batch_size_opt,
        batch_size_max,
        height_min,
        height_opt,
        height_max,
        width_min,
        width_opt,
        width_max,
        context_min,
        context_opt,
        context_max,
        num_video_frames,
    ):
        return super()._convert(
            model,
            filename_prefix,
            batch_size_min,
            batch_size_opt,
            batch_size_max,
            height_min,
            height_opt,
            height_max,
            width_min,
            width_opt,
            width_max,
            context_min,
            context_opt,
            context_max,
            num_video_frames,
            is_static=False,
        )


class STATIC_TRT_MODEL_CONVERSION(TRT_MODEL_CONVERSION_BASE):
    def __init__(self):
        super(STATIC_TRT_MODEL_CONVERSION, self).__init__()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "filename_prefix": ("STRING", {"default": "tensorrt/ComfyUI_STAT"}),
                "batch_size_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "height_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "width_opt": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "context_opt": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                    },
                ),
                "num_video_frames": (
                    "INT",
                    {
                        "default": 14,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                    },
                ),
            },
        }

    def convert(
        self,
        model,
        filename_prefix,
        batch_size_opt,
        height_opt,
        width_opt,
        context_opt,
        num_video_frames,
    ):
        return super()._convert(
            model,
            filename_prefix,
            batch_size_opt,
            batch_size_opt,
            batch_size_opt,
            height_opt,
            height_opt,
            height_opt,
            width_opt,
            width_opt,
            width_opt,
            context_opt,
            context_opt,
            context_opt,
            num_video_frames,
            is_static=True,
        )


NODE_CLASS_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
    "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION,
}
