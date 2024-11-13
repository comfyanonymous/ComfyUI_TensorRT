import torch
from comfy.supported_models import (
    SD15,
    SD20,
    SD21UnclipL,
    SD21UnclipH,
    SDXLRefiner,
    SDXL,
    SSD1B,
    Segmind_Vega,
    KOALA_700M,
    KOALA_1B,
    SVD_img2vid,
    SD15_instructpix2pix,
    SDXL_instructpix2pix,
)

from .baseline import TRTModelUtil


class UNetTRT(TRTModelUtil):
    def __init__(
            self,
            context_dim: int,
            input_channels: int,
            y_dim: int,
            hidden_size: int,
            channel_mult: tuple[int],
            num_res_blocks: tuple[int],
            context_len: int = 77,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(context_dim, input_channels, context_len, *args, **kwargs)

        self.hidden_size = hidden_size
        self.y_dim = y_dim
        self.is_conditional = True

        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.input_block_chans = self.set_block_chans()

        if self.y_dim:
            self.input_config.update({"y": {"batch": "{batch_size}", "y_dim": y_dim}})

        if self.use_control:
            self.control = self.get_control()
            self.input_config.update(self.control)

    def set_block_chans(self):
        ch = self.hidden_size
        ds = 1

        input_block_chans = [(self.hidden_size, ds)]
        for level, mult in enumerate(self.channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * self.hidden_size
                input_block_chans.append((ch, ds))
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                ch = out_ch
                ds *= 2
                input_block_chans.append((ch, ds))
        return input_block_chans

    @classmethod
    def from_model(cls, model, **kwargs):
        return cls(
            context_dim=model.model.model_config.unet_config["context_dim"],
            input_channels=model.model.diffusion_model.in_channels,
            hidden_size=model.model.model_config.unet_config["model_channels"],
            y_dim=model.model.model_config.unet_config.get("adm_in_channels", 0),
            channel_mult=model.model.diffusion_model.channel_mult,
            num_res_blocks=model.model.diffusion_model.num_res_blocks,
            **kwargs,
        )

    def to_dict(self):
        return {
            "context_dim": self.context_dim,
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "y_dim": self.y_dim,
            "channel_mult": self.channel_mult,
            "num_res_blocks": self.num_res_blocks,
            "use_control": self.use_control,
        }

    def get_control(self):
        control_input = {}

        for i, (ch, d) in enumerate(reversed(self.input_block_chans)):
            control_input[f"input_control_{i}"] = {
                "batch": "{batch_size}",
                "chn": ch,
                f"height{d}": "{height}//(8*" + str(d) + ")",
                f"width{d}": "{width}//(8*" + str(d) + ")",
            }

        for i, (ch, d) in enumerate(self.input_block_chans):
            control_input[f"output_control_{i}"] = {
                "batch": "{batch_size}",
                "chn": ch,
                f"height{d}": "{height}//(8*" + str(d) + ")",
                f"width{d}": "{width}//(8*" + str(d) + ")",
            }

        ch, d = self.input_block_chans[-1]
        control_input["middle_control_0"] = {
            "batch": "{batch_size}",
            "chn": ch,
            f"height{d}": "{height}//(8*" + str(d) + ")",
            f"width{d}": "{width}//(8*" + str(d) + ")",
        }
        return control_input

    def get_dtype(self):
        return torch.float16


class SD15_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SD15.unet_config["context_dim"],
            input_channels=4,
            y_dim=0,
            hidden_size=SD15.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )

    @classmethod
    def from_model(cls, model, **kwargs):
        return super(SD15_TRT, cls).from_model(model, use_control=True)


class SD20_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SD20.unet_config["context_dim"],
            input_channels=4,
            y_dim=0,
            hidden_size=SD20.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )

    @classmethod
    def from_model(cls, model, **kwargs):
        return super(SD20_TRT, cls).from_model(model, use_control=True)


class SD21UnclipL_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SD21UnclipL.unet_config["context_dim"],
            input_channels=4,
            y_dim=SD21UnclipL.unet_config["adm_in_channels"],
            hidden_size=SD21UnclipL.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class SD21UnclipH_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SD21UnclipH.unet_config["context_dim"],
            input_channels=4,
            y_dim=SD21UnclipH.unet_config["adm_in_channels"],
            hidden_size=SD21UnclipH.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class SDXLRefiner_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SDXLRefiner.unet_config["context_dim"],
            input_channels=4,
            y_dim=SDXLRefiner.unet_config["adm_in_channels"],
            hidden_size=SDXLRefiner.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class SDXL_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SDXL.unet_config["context_dim"],
            input_channels=4,
            y_dim=SDXL.unet_config["adm_in_channels"],
            hidden_size=SDXL.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )

    @classmethod
    def from_model(cls, model, **kwargs):
        return super(SDXL_TRT, cls).from_model(model, use_control=True)


class SSD1B_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SSD1B.unet_config["context_dim"],
            input_channels=4,
            y_dim=SSD1B.unet_config["adm_in_channels"],
            hidden_size=SSD1B.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class Segmind_Vega_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=Segmind_Vega.unet_config["context_dim"],
            input_channels=4,
            y_dim=Segmind_Vega.unet_config["adm_in_channels"],
            hidden_size=Segmind_Vega.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),  # TODO
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class KOALA_700M_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=KOALA_700M.unet_config["context_dim"],
            input_channels=4,
            y_dim=KOALA_700M.unet_config["adm_in_channels"],
            hidden_size=KOALA_700M.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),  # TODO
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class KOALA_1B_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=KOALA_1B.unet_config["context_dim"],
            input_channels=4,
            y_dim=KOALA_1B.unet_config["adm_in_channels"],
            hidden_size=KOALA_1B.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),  # TODO
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class SVD_img2vid_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SVD_img2vid.unet_config["context_dim"],
            input_channels=8,
            y_dim=SVD_img2vid.unet_config["adm_in_channels"],
            hidden_size=SVD_img2vid.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )

        self.input_config["context"]["context_len"] = self.context_len


class SD15_instructpix2pix_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SD15_instructpix2pix.unet_config["context_dim"],
            input_channels=8,
            y_dim=0,
            hidden_size=SD15_instructpix2pix.unet_config["model_channels"],
            channel_mult=(1, 2, 4, 4),
            num_res_blocks=(2, 2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )


class SDXL_instructpix2pix_TRT(UNetTRT):
    def __init__(
            self,
            context_dim=SDXL_instructpix2pix.unet_config["context_dim"],
            input_channels=8,
            y_dim=SDXL_instructpix2pix.unet_config["adm_in_channels"],
            hidden_size=SDXL_instructpix2pix.unet_config["model_channels"],
            channel_mult=(1, 2, 4),
            num_res_blocks=(2, 2, 2),
            **kwargs,
    ):
        super().__init__(
            context_dim,
            input_channels,
            y_dim,
            hidden_size,
            channel_mult,
            num_res_blocks,
            **kwargs,
        )
