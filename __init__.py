from .tensorrt_nodes import NODE_CLASS_MAPPINGS as TRT_CLASS_MAP
from .tensorrt_nodes import NODE_DISPLAY_NAME_MAPPINGS as TRT_NAME_MAP

from .onnx_nodes import NODE_CLASS_MAPPING as ONNX_CLASS_MAP
from .onnx_nodes import NODE_DISPLAY_NAME_MAPPINGS as ONNX_NAME_MAP

NODE_CLASS_MAPPINGS = TRT_CLASS_MAP | ONNX_CLASS_MAP
NODE_DISPLAY_NAME_MAPPINGS = TRT_NAME_MAP | ONNX_NAME_MAP

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
