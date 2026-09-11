from model_loaders.Qwen_VL import Qwen2_5_VL

class VLM(Qwen2_5_VL):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="Qwen2.5-VL-3B-Instruct-AWQ",
            **kwargs
        )