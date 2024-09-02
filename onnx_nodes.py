import os
from .onnx_utils.export import export_onnx
import comfy 

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
                "output_folder": ("STRING",)
            },
            "optional": {
                "filename": ("STRING", {"default": "model.onnx"})
            }
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


NODE_CLASS_MAPPING = {
    "ONNX_EXPORT": ONNX_EXPORT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ONNX_EXPORT": "ONNX Export",
}