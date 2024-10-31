from .baseline import TRTModelUtil
from .flux import Flux_TRT, FluxSchnell_TRT
from .auraflow import AuraFlow_TRT
from .sd3 import SD3_TRT
from .sd_unet import (
    SD15_TRT,
    SD20_TRT,
    SD21UnclipH_TRT,
    SD21UnclipL_TRT,
    SDXL_instructpix2pix_TRT,
    SDXLRefiner_TRT,
    SDXL_TRT,
    SSD1B_TRT,
    KOALA_700M_TRT,
    KOALA_1B_TRT,
    Segmind_Vega_TRT,
    SVD_img2vid_TRT,
)
import comfy.supported_models

supported_models = {
    "SD15": SD15_TRT,
    "SD20": SD20_TRT,
    "SD21UnclipL": SD21UnclipL_TRT,
    "SD21UnclipH": SD21UnclipH_TRT,
    "SDXL_instructpix2pix": SDXL_instructpix2pix_TRT,
    "SDXLRefiner": SDXLRefiner_TRT,
    "SDXL": SDXL_TRT,
    "SSD1B": SSD1B_TRT,
    "KOALA_700M": KOALA_700M_TRT,
    "KOALA_1B": KOALA_1B_TRT,
    "Segmind_Vega": Segmind_Vega_TRT,
    "SVD_img2vid": SVD_img2vid_TRT,
    "SD3": SD3_TRT,
    "AuraFlow": AuraFlow_TRT,
    "Flux": Flux_TRT,
    "FluxSchnell": FluxSchnell_TRT,
}

unsupported_models = [
    "SV3D_u",
    "SV3D_p",
    "Stable_Zero123",
    "SD_X4Upscaler",
    "Stable_Cascade_C",
    "Stable_Cascade_B",
    "StableAudio",
    "HunyuanDiT",
    "HunyuanDiT1",
]


def detect_version_from_model(model):
    return model.model.model_config.__class__.__name__


def get_helper_from_version(model_version: str, config: dict = {}) -> TRTModelUtil:
    model_helper = supported_models.get(model_version, None)
    if model_helper is None:
        raise NotImplementedError("{} is not supported.".format(model_version))
    return model_helper(**config)


def get_helper_from_model(model) -> TRTModelUtil:
    model_version = detect_version_from_model(model)
    helper_cls = supported_models.get(model_version, None)
    if helper_cls is None:
        raise NotImplementedError("{} is not supported.".format(model_version))
    return helper_cls.from_model(model)


def get_model_from_version(model_version: str, config: dict = {}):
    conf = getattr(comfy.supported_models, model_version)
    helper = get_helper_from_version(model_version, config)
    conf.unet_config["disable_unet_model_creation"] = True
    conf.unet_config["in_channels"] = helper.input_channels
    conf = conf(conf.unet_config)
    if model_version in ("SD20",):
        model = comfy.model_base.BaseModel(
            conf, model_type=comfy.model_base.ModelType.V_PREDICTION
        )
    else:
        model = conf.get_model({})
    return model, helper
