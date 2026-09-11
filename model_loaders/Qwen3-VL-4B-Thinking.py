from model_loaders.Qwen_VL import Qwen3_VL

class VLM(Qwen3_VL):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="Qwen3-VL-4B-Thinking",
            **kwargs
        )