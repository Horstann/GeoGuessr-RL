from model_loaders.Qwen2_5_VL import Qwen2_5_VL

class VLM(Qwen2_5_VL):
    def __init__(self, **kwargs):
        super().__init__(
            size="7B",
            **kwargs
        )