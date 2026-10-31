from pathlib import Path
from importlib import import_module
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import yaml

logger = logging.getLogger("RRB_logger")
if not logger.hasHandlers():
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def get_evaluator(env_name: str, traj_type: str = "pre-defined"):
    """Resolve the maintained evaluator for the selected action representation."""
    if env_name != "rrb-man":
        raise ValueError(f"Unknown environment: {env_name}")
    if traj_type in ("pre-defined", "point"):
        module_name = "rrb_manipulation_evaluator_with_general_planner"
    elif traj_type == "code":
        module_name = "rrb_manipulation_evaluator_with_general_planner_code_generation"
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    module = import_module(f"responsible_robot_bench.evaluator.{module_name}")
    return module.RRB_ManipulationEvaluator

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.getLogger().handlers.clear()
    
    if 'log_level' not in cfg or cfg.log_level == "INFO":
        logger.setLevel(logging.INFO)
    if 'log_level' in cfg and cfg.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)

    env_name = cfg.env
    logger.info(f"Evaluating environment: {env_name}")
    
    if env_name != "rrb-man":
        raise ValueError(f"Unknown environment: {env_name}")
    with (Path(__file__).resolve().parent / "configs" / f"{env_name}.yaml").open() as f:
        base_config = yaml.safe_load(f)

    override_config = {
        k: v for k, v in OmegaConf.to_container(cfg).items() 
        if k != 'env' and v is not None
    }
    
    config = OmegaConf.merge(
        OmegaConf.create(base_config),
        override_config
    )

    print(config)
    logger.info("Starting evaluation")
    evaluator_class = get_evaluator(env_name, config["traj_type"])
    evaluator = evaluator_class(OmegaConf.to_container(config, resolve=True))
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
