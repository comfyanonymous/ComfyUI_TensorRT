from .baseline import TRTModelUtil


class AuraFlow_TRT(TRTModelUtil):
    def __init__(
            self, context_dim=2048, input_channels=4, context_len=256, **kwargs
    ) -> None:
        super().__init__(
            context_dim=context_dim,
            input_channels=input_channels,
            context_len=context_len,
            **kwargs,
        )
        self.is_conditional = True

    @classmethod
    def from_model(cls, model, **kwargs):
        return cls(
            context_dim=model.model.model_config.unet_config["cond_seq_dim"],
            input_channels=model.model.diffusion_model.out_channels,
            use_control=False,
            **kwargs,
        )
