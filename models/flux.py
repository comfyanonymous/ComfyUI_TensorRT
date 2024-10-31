from .baseline import TRTModelUtil
import torch


class FLuxBase(TRTModelUtil):
    def __init__(
        self,
        context_dim: int,
        input_channels: int,
        y_dim: int,
        hidden_size: int,
        double_blocks: int,
        single_blocks: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(context_dim, input_channels, 256, *args, **kwargs)

        self.hidden_size = hidden_size
        self.y_dim = y_dim
        self.single_blocks = single_blocks
        self.double_blocks = double_blocks

        self.extra_input = {
            "guidance": {"batch": "{batch_size}"},
            "y": {"batch": "{batch_size}", "y_dim": y_dim},
        }

        self.input_config.update(self.extra_input)

        if self.use_control:
            self.control = self.get_control(double_blocks, single_blocks)
            self.input_config.update(self.control)

    def to_dict(self):
        return {
            self.__name__: {
                "context_dim": self.context_dim,
                "input_channels": self.input_channels,
                "y_dim": self.y_dim,
                "hidden_size": self.hidden_size,
                "double_blocks": self.double_blocks,
                "single_blocks": self.single_blocks,
                "use_control": self.use_control,
            }
        }

    def get_control(self, double_blocks: int, single_blocks: int):
        control_input = {}
        for i in range(double_blocks):
            control_input[f"input_control_{i}"] = {
                "batch": "{batch_size}",
                "ids": "({height}*{width}//(8*2)**2)",
                "hidden_size": self.hidden_size,
            }
        for i in range(single_blocks):
            control_input[f"output_control_{i}"] = {
                "batch": "{batch_size}",
                "ids": "({height}*{width}//(8*2)**2)",
                "hidden_size": self.hidden_size,
            }
        return control_input

    def get_dtype(self):
        return torch.bfloat16

    @classmethod
    def from_model(cls, model, **kwargs):
        return cls(
            context_dim=model.model.model_config.unet_config["context_in_dim"],
            input_channels=model.model.model_config.unet_config["in_channels"],
            hidden_size=model.model.model_config.unet_config["hidden_size"],
            y_dim=model.model.model_config.unet_config["vec_in_dim"],
            double_blocks=model.model.model_config.unet_config["depth"],
            single_blocks=model.model.model_config.unet_config["depth_single_blocks"],
            **kwargs,
        )


class Flux_TRT(FLuxBase):
    def __init__(
        self,
        context_dim=4096,
        input_channels=16,
        y_dim=768,
        hidden_size=3072,
        double_blocks=19,
        single_blocks=28,
        **kwargs,
    ):
        super().__init__(
            context_dim=context_dim,
            input_channels=input_channels,
            y_dim=y_dim,
            hidden_size=hidden_size,
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            **kwargs,
        )

    @classmethod
    def from_model(cls, model):
        return super(Flux_TRT, cls).from_model(model, use_control=True)


class FluxSchnell_TRT(FLuxBase):
    def __init__(
        self,
        context_dim=4096,
        input_channels=16,
        y_dim=768,
        hidden_size=3072,
        double_blocks=19,
        single_blocks=28,
        **kwargs,
    ):
        super().__init__(
            context_dim=context_dim,
            input_channels=input_channels,
            y_dim=y_dim,
            hidden_size=hidden_size,
            double_blocks=double_blocks,
            single_blocks=single_blocks,
            **kwargs,
        )

    @classmethod
    def from_model(cls, model):
        return super(FluxSchnell_TRT, cls).from_model(model, use_control=True)
