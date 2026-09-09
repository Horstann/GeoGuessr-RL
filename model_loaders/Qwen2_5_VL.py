"""Standalone local Qwen2.5-VL backend (7B, 32B, or 72B).

Example after importing this file with importlib (its filename contains dots):
    backend = Qwen2_5_VL(size="7B")
    backend.load()  # Download if needed, then load into CPU RAM once.
    result = backend.generate([{"role": "user", "content": "Hello!"}])
    print(result.text)

Share this backend instance across graph workers. Creating another
instance loads another copy. This module does not wire itself into ai_client.py.
"""

from threading import RLock
import torch
from model_loaders.factory import MODELS_DIR, GenerationResult


class Qwen2_5_VL:
    """Own one model/processor pair; keep conversation state in the caller."""

    SUPPORTED_SIZES = ("7B", "32B", "72B")
    MODELS_DIR = MODELS_DIR

    def __init__(self, size="7B", dtype="auto"):
        size = str(size).upper()
        if size not in self.SUPPORTED_SIZES:
            raise ValueError(f"size must be one of {self.SUPPORTED_SIZES}, got {size!r}")
        self.repo_id = f"Qwen/Qwen2.5-VL-{size}-Instruct"
        self.model_path = self.MODELS_DIR / self.repo_id.split("/")[-1]
        self.device = (
            "cuda" if torch.cuda.is_available() else 
            # "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._lock = RLock()

    @property
    def is_loaded(self):
        return self.model is not None and self.processor is not None

    def download(self):
        """Fetch weights/configs; Hub metadata reuses already downloaded files.

        This can contact Hugging Face to check the snapshot, but unchanged
        files are not downloaded again. Interrupted downloads can be resumed
        by calling this method again. Use load(local_files_only=True) offline.
        Authentication, if needed, uses the standard HF_TOKEN environment variable.
        """
        from huggingface_hub import snapshot_download

        with self._lock:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(self.model_path),
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
            )
        return self.model_path

    def load(self, *, local_files_only=False):
        """Load once into RAM (default) or the configured device; return self.

        local_files_only=True skips downloading and fails if local files are
        missing or incomplete. Loading larger variants requires enough memory.
        """
        with self._lock:
            if self.is_loaded:
                return self
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            if not local_files_only:
                self.download()
            processor = AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                device_map=self.device,
                dtype=self.dtype,
                local_files_only=True,
            )
            model.eval() # .eval() for inference mode (vs .train() for training mode)
            # affects layers like Dropout, BatchNorm etc
            self.processor = processor
            self.model = model
        return self

    @staticmethod
    def prepare_messages(messages):
        """Convert OpenAI text/image blocks to Qwen blocks without mutating input.

        Supports image_url URLs/data URIs and native Qwen image blocks with
        local paths or PIL images. Video and tool messages are not implemented.
        """
        prepared = []
        for message in messages:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported role: {message['role']!r}")
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            blocks = []
            for block in content:
                if block["type"] == "image_url":
                    blocks.append({"type": "image", "image": block["image_url"]["url"]})
                elif block["type"] in {"text", "image"}:
                    blocks.append(dict(block))
                else:
                    raise ValueError(f"Unsupported content type: {block['type']!r}")
            prepared.append({"role": message["role"], "content": blocks})
        return prepared

    def generate(self, messages, *, max_tokens=512, temperature=0.0, num_beams=1):
        """Generate one response. Call load() before submitting graph workers.

        A single lock serializes inference and prevents unload during generation.
        Returns a backend result, not an OpenAI chat-completion response.
        """
        from qwen_vl_utils import process_vision_info

        max_tokens = 512 if max_tokens is None else max_tokens
        with self._lock:
            if not self.is_loaded:
                raise RuntimeError("Call load() before generate().")
            messages = self.prepare_messages(messages)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            images, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, padding=True, return_tensors="pt",
            ).to(self.model.device)
            options = {
                "max_new_tokens": max_tokens, 
                "num_beams": num_beams,
            }
            if temperature > 0:
                options["temperature"] = temperature
            else:
                options["do_sample"] = False
            with torch.inference_mode(): # disables gradient tracking and additional bookkeeping, reducing memory use and execution overhead.
                output = self.model.generate(**inputs, **options)
            input_tokens = inputs["input_ids"].shape[1]
            generated = output[0, input_tokens:]
            answer = self.processor.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            return GenerationResult(answer, input_tokens, len(generated))

    def unload(self):
        """Release this backend's references; keep downloaded files on disk.

        Accelerator allocators may retain freed memory for reuse.
        """
        with self._lock:
            self.model = None
            self.processor = None
