import logging
import logging.config
import sys, os
from omegaconf import OmegaConf

def quiet_vllm_logger():
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_LOG_LEVEL"] = "WARNING"


def apply_overrides(config):
    base = OmegaConf.structured(config)
    
    # Get CLI args up to '--' if present, otherwise all args
    args = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    cli_args = [arg.lstrip("-") for arg in args]
    
    # Merge overrides
    overrides = OmegaConf.from_cli(cli_args)
    merged = OmegaConf.merge(base, overrides)
    return OmegaConf.to_object(merged)