import hydra
from omegaconf import DictConfig
from geo_aot_geoguess import GeoAoTGuesser
import json
import os
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from batch_process_aot import main as batch_main
    #  cfg.output_folder = os.path.join(cfg.output_folder, cfg.ai_config.model)
    input_folder = getattr(cfg.ai_config, 'input_graphs_folder', None)

    # Now check if the value is "truthy" (i.e., not None, not an empty string, etc.)
    if input_folder:
        cfg.batch_process.input_graphs_folder = input_folder
    batch_main(cfg)

if __name__ == "__main__":
    main()
