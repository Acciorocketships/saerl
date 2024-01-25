#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl
from benchmarl.hydra_config import load_experiment_from_hydra
from saerl.models.sae_model import SAEModelConfig
from saerl.models.vanilla_model import VanillaModelConfig
from saerl.models.deepset_model import DeepSetModelConfig

def update_registries():
    benchmarl.models.model_config_registry.update({
        "sae_model": SAEModelConfig,
        "vanilla_model": VanillaModelConfig,
        "deepset_model": DeepSetModelConfig,
    })

@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    """Runs an experiment loading its config from hydra.

    This function is decorated as ``@hydra.main`` and is called by running

    .. code-block:: console

       python benchmarl/run.py algorithm=mappo task=vmas/balance


    Args:
        cfg (DictConfig): the hydra config dictionary

    """
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name)
    experiment.run()


if __name__ == "__main__":
    update_registries()
    hydra_experiment()
