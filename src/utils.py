import logging
import sys
from omegaconf import OmegaConf

def quiet_vllm_logger(level=logging.WARNING):
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith("vllm"):
            logging.getLogger(name).setLevel(level)


def apply_overrides(config):
    base = OmegaConf.structured(config)
    
    # Get CLI args up to '--' if present, otherwise all args
    args = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    cli_args = [arg.lstrip("-") for arg in args]
    
    # Merge overrides
    overrides = OmegaConf.from_cli(cli_args)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_object(merged)