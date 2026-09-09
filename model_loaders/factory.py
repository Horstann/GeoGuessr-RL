import sys
from pathlib import Path
import importlib.util
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

class GenerationResult(NamedTuple):
    text: str
    input_tokens: int
    output_tokens: int

def load_vlm(ai_config):
    """Import the configured loader by file path and load one shared model."""
    loader_path = ai_config.get('loader_path')
    if not loader_path:
        return None
    path = Path(loader_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Model loader not found: {path}")

    # Import from dynamic path
    spec = importlib.util.spec_from_file_location('vlm_loader', path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model loader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise

    kwargs = {
        name: ai_config[name]
        for name in ('dtype',)
        if ai_config.get(name) is not None
    }
    vlm = module.VLM(**kwargs)
    print(f"Loading shared VLM from {path}")
    try:
        vlm.load(local_files_only=ai_config.get('local_files_only', False))
    except BaseException:
        vlm.unload()
        raise
    return vlm
