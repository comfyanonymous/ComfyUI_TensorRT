from .baseline import TRTModelUtil
import torch


class SD3_TRT(TRTModelUtil):
    def __init__(
        self,
        context_dim: int = 4096,
        input_channels: int = 16,
        y_dim: int = 2048,
        hidden_size: int = 1536,
        output_blocks: int = 24,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(context_dim, input_channels, 77, *args, **kwargs)

        self.hidden_size = hidden_size
        self.y_dim = y_dim
        self.is_conditional = True
        self.output_blocks = output_blocks  # - 1 # self.joint_blocks

        self.extra_input = {
            "y": {"batch": "{batch_size}", "y_dim": y_dim},
        }

        self.input_config.update(self.extra_input)

        if self.use_control:
            self.control = self.get_control()
            self.input_config.update(self.control)

    def to_dict(self):
        return {
            "context_dim": self.context_dim,
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "y_dim": self.y_dim,
            "output_blocks": self.output_blocks,
            "use_control": self.use_control,
        }

    def get_control(self):
        control_input = {}
        for i in range(self.output_blocks):
            control_input[f"output_control_{i}"] = {
                "batch": "{batch_size}",
                "ids": "({height}*{width}//(8*2)**2)",
                "hidden_size": self.hidden_size,
            }

        return control_input

    def get_dtype(self):
        return torch.float16

    @classmethod
    def from_model(cls, model, **kwargs):
        return cls(
            context_dim=model.model.diffusion_model.context_embedder.in_features,
            input_channels=model.model.diffusion_model.in_channels,
            hidden_size=model.model.diffusion_model.context_embedder.out_features,
            y_dim=model.model.model_config.unet_config.get("adm_in_channels", 0),
            output_blocks=model.model.diffusion_model.depth,
            use_control=True,
            **kwargs,
        )
