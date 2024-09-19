import os
from .onnx_utils.export import export_onnx
import comfy
import folder_paths


class ONNX_EXPORT:
    def __init__(self) -> None:
        pass

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "output_folder": (
                    "STRING",
                    {"default": os.path.join(folder_paths.models_dir, "onnx")},
                ),
            },
            "optional": {"filename": ("STRING", {"default": "model.onnx"})},
        }

    def export(self, model, output_folder, filename):
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        path = os.path.join(output_folder, filename)
        export_onnx(model, path)
        print(f"INFO: Exported Model to: {path}")
        return ()


class ONNXModelSelector:
    @classmethod
    def INPUT_TYPES(s):
        onnx_path = os.path.join(folder_paths.models_dir, "onnx")
        if not os.path.exists(onnx_path):
            os.makedirs(onnx_path)
        onnx_models = [f for f in os.listdir(onnx_path) if f.endswith(".onnx")]
        return {
            "required": {
                "model_name": (onnx_models,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "model_name")
    FUNCTION = "select_onnx_model"
    CATEGORY = "TensorRT"

    def select_onnx_model(self, model_name):
        onnx_path = os.path.join(folder_paths.models_dir, "onnx")
        model_path = os.path.join(onnx_path, model_name)
        return (model_path, model_name)


NODE_CLASS_MAPPING = {
    "ONNX_EXPORT": ONNX_EXPORT,
    "ONNXModelSelector": ONNXModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ONNX_EXPORT": "ONNX Export",
    "ONNXModelSelector": "Select ONNX Model",
}
