# Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)
import os
import time
from typing import Optional

import comfy.model_management
import comfy.model_patcher
import folder_paths

from .models import supported_models, detect_version_from_model, get_helper_from_model
from .onnx_utils.export import export_onnx
from .tensorrt_diffusion_model import TRTDiffusionBackbone

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


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("tensorrt"),),
                "model_type": (list(supported_models.keys()),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    @staticmethod
    def load_unet(unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        model = TRTDiffusionBackbone.load_trt_model(unet_path, model_type)
        return (
            comfy.model_patcher.ModelPatcher(
                model,
                load_device=comfy.model_management.get_torch_device(),
                offload_device=comfy.model_management.unet_offload_device(),
            ),
        )


class TRTBuildBase:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.timing_cache_path = os.path.normpath(
            os.path.join(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "timing_cache.trt"
                )
            )
        )

    RETURN_TYPES = ()
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(s):
        raise NotImplementedError

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
            output_onnx: Optional[str] = None,
    ):
        if output_onnx is None:
            output_onnx = os.path.normpath(
                os.path.join(
                    os.path.join(self.temp_dir, "{}".format(time.time())), "model.onnx"
                )
            )
            os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

            comfy.model_management.unload_all_models()
            comfy.model_management.load_models_gpu(
                [model], force_patch_weights=True, force_full_load=True
            )
            export_onnx(model, output_onnx)

        model_version = detect_version_from_model(model)
        model_helper = get_helper_from_model(model)
        trt_model = TRTDiffusionBackbone(model_helper)

        filename_prefix = f"{filename_prefix}_{model_version}"
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

        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        )
        output_trt_engine = os.path.join(
            full_output_folder, f"{filename}_{counter:05}_.engine"
        )

        batch_multiplier = (
            2 if model_helper.is_conditional else 1
        )  # TODO lets see if we really want this
        if model_version == "SVD_img2vid":
            batch_multiplier *= num_video_frames
        success = trt_model.build(
            output_onnx,
            output_trt_engine,
            self.timing_cache_path,
            opt_config={
                "batch_size": batch_size_opt * batch_multiplier,
                "height": height_opt,
                "width": width_opt,
                "context_len": context_opt * model_helper.context_len,
            },
            min_config={
                "batch_size": batch_size_min * batch_multiplier,
                "height": height_min,
                "width": width_min,
                "context_len": context_min * model_helper.context_len,
            },
            max_config={
                "batch_size": batch_size_max * batch_multiplier,
                "height": height_max,
                "width": width_max,
                "context_len": context_max * model_helper.context_len,
            },
        )
        if not success:
            raise RuntimeError("Engine Build Failed")
        return ()


class DynamicTRTBuild(TRTBuildBase):
    def __init__(self):
        super(DynamicTRTBuild, self).__init__()

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
            "optional": {
                "onnx_model_path": ("STRING", {"default": "", "forceInput": True}),
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
            onnx_model_path,
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
            output_onnx=onnx_model_path,
        )


class StaticTRTBuild(TRTBuildBase):
    def __init__(self):
        super(StaticTRTBuild, self).__init__()

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
            "optional": {
                "onnx_model_path": ("STRING", {"default": "", "forceInput": True}),
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
            onnx_model_path,
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
            output_onnx=onnx_model_path,
        )


NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
    "DYNAMIC_TRT_MODEL_CONVERSION": DynamicTRTBuild,
    "STATIC_TRT_MODEL_CONVERSION": StaticTRTBuild,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTLoader": "TensorRT Loader",
    "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION",
    "STATIC TRT_MODEL CONVERSION": "STATIC_TRT_MODEL_CONVERSION",
}
