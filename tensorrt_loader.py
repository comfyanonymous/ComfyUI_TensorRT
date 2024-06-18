#Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import torch
import os

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine"})

import tensorrt as trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)

# Is there a function that already exists for this?
def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16

class TrTUnet:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.dtype = torch.float16

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        #Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        #for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape, 
                          device=x.device, 
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(k, x[(x.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        # stream.synchronize() #don't need to sync stream since it's the default torch one
        return out

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("tensorrt"), ),
                             "model_type": (["sdxl_base", "sdxl_refiner", "sd1.x", "sd2.x-768v", "svd", "sd3"], ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
        elif model_type == "sdxl_refiner":
            conf = comfy.supported_models.SDXLRefiner(
                {"adm_in_channels": 2560})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXLRefiner(conf)
        elif model_type == "sd1.x":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd2.x-768v":
            conf = comfy.supported_models.SD20({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf, model_type=comfy.model_base.ModelType.V_PREDICTION)
        elif model_type == "svd":
            conf = comfy.supported_models.SVD_img2vid({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "sd3":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        model.diffusion_model = unet
        model.memory_required = lambda *args, **kwargs: 0 #always pass inputs batched up as much as possible, our TRT code will handle batch splitting

        return (comfy.model_patcher.ModelPatcher(model,
                                                 load_device=comfy.model_management.get_torch_device(),
                                                 offload_device=comfy.model_management.unet_offload_device()),)

NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
}