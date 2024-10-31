import torch


class TRTModelUtil:
    def __init__(
        self,
        context_dim: int,
        input_channels: int,
        context_len: int,
        use_control: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.context_dim = context_dim
        self.input_channels = input_channels
        self.context_len = context_len
        self.use_control = use_control
        self.is_conditional = False

        self.input_config = {
            "x": {
                "batch": "{batch_size}",
                "input_channels": self.input_channels,
                "height": "{height}//8",
                "width": "{width}//8",
            },
            "timesteps": {
                "batch": "{batch_size}",
            },
            "context": {
                "batch": "{batch_size}",
                "context_len": "{context_len}",
                "context_dim": self.context_dim,
            },
        }

        self.output_config = {
            "h": {
                "batch": "{batch_size}",
                "input_channels": self.input_channels,
                "height": "{height}//8",
                "width": "{width}//8",
            }
        }

    def to_dict(self):
        return {
            "context_dim": self.context_dim,
            "input_channels": self.input_channels,
            "context_len": self.context_dim,
            "use_control": self.use_control,
        }

    def get_input_names(self) -> list[str]:
        return list(self.input_config.keys())

    def get_output_names(self) -> list[str]:
        return list(self.output_config.keys())

    def get_dtype(self) -> torch.dtype:
        return torch.float16

    def get_input_shapes(self, **kwargs) -> dict:
        inputs_shapes = {}
        for io_name, io_config in self.input_config.items():
            _inp = self._eval_shape(io_config, **kwargs)
            inputs_shapes[io_name] = _inp

        return inputs_shapes

    def get_input_shapes_by_key(self, key: str, **kwargs) -> tuple[int]:
        return self._eval_shape(self.input_config[key], **kwargs)

    def get_dynamic_axes(self, config: dict = {}) -> dict:
        dynamic_axes = {}

        if config == {}:
            config = self.input_config | self.output_config
        for k, v in config.items():
            dyn = {i: ax for i, (ax, s) in enumerate(v.items()) if isinstance(s, str)}
            dynamic_axes[k] = dyn

        return dynamic_axes

    def _eval_shape(self, inp, **kwargs) -> tuple[int]:
        if "context_len" not in kwargs:
            kwargs["context_len"] = self.context_len
        shape = []
        for _, v in inp.items():
            _s = v
            if isinstance(v, str):
                _s = int(eval(v.format(**kwargs)))
            shape.append(_s)
        return tuple(shape)

    def get_control(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    @classmethod
    def from_model(cls, model, **kwargs):
        raise NotImplementedError
