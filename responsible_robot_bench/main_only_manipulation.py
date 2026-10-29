import os
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

class_names = {"rrb-man": "RRB_ManipulationEvaluator"}
module_names = {"rrb-man": "rrb_manipulation_evaluator_with_planner"}

def get_evaluator(env_name: str):

    if env_name not in module_names:
        raise ValueError(f"Unknown environment: {env_name}")
    
    module_name = f"responsible_robot_bench.evaluator.{module_names[env_name]}"
    evaluator_name = class_names[env_name]
    
    module = __import__(module_name, fromlist=[evaluator_name])
    return getattr(module, evaluator_name)

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.getLogger().handlers.clear()
    
    if 'log_level' not in cfg or cfg.log_level == "INFO":
        logger.setLevel(logging.INFO)
    if 'log_level' in cfg and cfg.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)

    env_name = cfg.env
    logger.info(f"Evaluating environment: {env_name}")
    
    with open(f"responsible_robot_bench/configs/{env_name}.yaml", 'r') as f:
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
    evaluator_class = get_evaluator(env_name)
    evaluator = evaluator_class(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main()
