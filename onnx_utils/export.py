import json
import os
import time
from typing import List

import comfy
import folder_paths
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnx.external_data_helper import _get_all_tensors, ExternalDataInfo

from ..models import detect_version_from_model, get_helper_from_model


def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> List[str]:
    """
    Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_all_tensors(model)
    model_tensors_ext = [
        ExternalDataInfo(tensor).location
        for tensor in model_tensors
        if tensor.HasField("data_location")
           and tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    return model_tensors_ext


def get_sample_input(input_shapes: dict, dtype: torch.dtype, device: torch.device):
    inputs = []
    for k, shape in input_shapes.items():
        inputs.append(
            torch.zeros(
                shape,
                device=device,
                dtype=dtype,
            )
        )

    return tuple(inputs)


def get_backbone(model, model_version, input_names, num_video_frames, use_control):
    unet = model.model.diffusion_model
    transformer_options = model.model_options["transformer_options"].copy()

    if model_version == "SVD_img2vid":

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
    else:

        class UNET(torch.nn.Module):
            def forward(self, x, timesteps, context, *args):
                extras = input_names[3:]
                control = {"input": [], "output": [], "middle": []}
                extra_args = {}
                for i in range(len(extras)):
                    if "control" in extras[i]:
                        if "input" in extras[i]:
                            control["input"].append(args[i])
                        elif "output" in extras[i]:
                            control["output"].append(args[i])
                        elif "middle" in extras[i]:
                            control["middle"].append(args[i])
                    else:
                        extra_args[extras[i]] = args[i]
                if use_control:
                    extra_args["control"] = control
                return self.unet(
                    x,
                    timesteps,
                    context,
                    transformer_options=self.transformer_options,
                    **extra_args,
                )

        _unet = UNET()
        _unet.unet = unet
        _unet.transformer_options = transformer_options
        unet = _unet

    return unet


# Helper utility for weights map
def export_weights_map(state_dict, onnx_opt_path: str, weights_map_path: str):
    onnx_opt_dir = onnx_opt_path
    onnx_opt_model = onnx.load(onnx_opt_path)

    # Create initializer data hashes
    def init_hash_map(onnx_opt_model):
        initializer_hash_mapping = {}
        for initializer in onnx_opt_model.graph.initializer:
            initializer_data = numpy_helper.to_array(
                initializer, base_dir=onnx_opt_dir
            ).astype(np.float16)
            initializer_hash = hash(initializer_data.data.tobytes())
            initializer_hash_mapping[initializer.name] = (
                initializer_hash,
                initializer_data.shape,
            )
        return initializer_hash_mapping

    initializer_hash_mapping = init_hash_map(onnx_opt_model)

    weights_name_mapping = {}
    weights_shape_mapping = {}
    # set to keep track of initializers already added to the name_mapping dict
    initializers_mapped = set()
    for wt_name, wt in state_dict.items():
        # get weight hash
        wt = wt.cpu().detach().numpy().astype(np.float16)
        wt_hash = hash(wt.data.tobytes())
        wt_t_hash = hash(np.transpose(wt).data.tobytes())

        for initializer_name, (
                initializer_hash,
                initializer_shape,
        ) in initializer_hash_mapping.items():
            # Due to constant folding, some weights are transposed during export
            # To account for the transpose op, we compare the initializer hash to the
            # hash for the weight and its transpose
            if wt_hash == initializer_hash or wt_t_hash == initializer_hash:
                # The assert below ensures there is a 1:1 mapping between
                # PyTorch and ONNX weight names. It can be removed in cases where 1:many
                # mapping is found and name_mapping[wt_name] = list()
                assert initializer_name not in initializers_mapped
                weights_name_mapping[wt_name] = initializer_name
                initializers_mapped.add(initializer_name)
                is_transpose = False if wt_hash == initializer_hash else True
                weights_shape_mapping[wt_name] = (
                    initializer_shape,
                    is_transpose,
                )

        # Sanity check: Were any weights not matched
        if wt_name not in weights_name_mapping:
            print(f"[I] PyTorch weight {wt_name} not matched with any ONNX initializer")
    print(
        f"[I] UNet: {len(weights_name_mapping.keys())} PyTorch weights were matched with ONNX initializers"
    )

    assert weights_name_mapping.keys() == weights_shape_mapping.keys()
    with open(weights_map_path, "w") as fp:
        json.dump([weights_name_mapping, weights_shape_mapping], fp)


def export_onnx(
        model,
        path,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        num_video_frames: int = 14,
        context_multiplier: int = 1,
):
    model_version = detect_version_from_model(model)
    model_helper = get_helper_from_model(model)

    dtype = model_helper.get_dtype()
    device = comfy.model_management.get_torch_device()
    input_names = model_helper.get_input_names()
    output_names = model_helper.get_output_names()
    dynamic_axes = model_helper.get_dynamic_axes()

    if model_version == "SVD_img2vid":
        batch_size *= num_video_frames
    if model_helper.is_conditional:
        batch_size *= 2
    input_shapes = model_helper.get_input_shapes(
        batch_size=batch_size,
        height=height,
        width=width,
        context_multiplier=context_multiplier,
    )
    inputs = get_sample_input(input_shapes, dtype, device)
    backbone = get_backbone(
        model, model_version, input_names, num_video_frames, model_helper.use_control
    )

    _, name = os.path.split(path)
    temp_path = os.path.join(
        folder_paths.get_temp_directory(), "{}".format(time.time())
    )
    onnx_temp = os.path.normpath(os.path.join(temp_path, name))

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    torch.onnx.export(
        backbone,
        inputs,
        onnx_temp,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    onnx_model = onnx.load(onnx_temp, load_external_data=True)
    tensors_paths = _get_onnx_external_data_tensors(onnx_model)

    if tensors_paths:
        for tensor in tensors_paths:
            os.remove(os.path.join(onnx_temp, tensor))

    onnx.save(
        onnx_model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=name + "_data",
        size_threshold=1024,
    )
