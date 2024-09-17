import numpy as np
import hydra
import omegaconf
import pickle
from pprint import pprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import logging
import os
from random import randint
import config
import scipy


@hydra.main(config_path="config",
            config_name="default_kp_config",
            version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:

    # Format the config file
    kp = hydra.utils.instantiate(cfg.pykoop_pipeline, _convert_='all')
    kp_true = hydra.utils.instantiate(cfg.pykoop_pipeline, _convert_='all')
    hydra_cfg = HydraConfig.get()
    # Get parameters and create folders
    if cfg.robot == 'nl_msd':
        path = "build/pykoop_objects/{}/variance_{}/kp_{}_{}.bin".format(
            cfg.robot, cfg.variance,
            hydra_cfg.runtime.choices['regressors@pykoop_pipeline'],
            hydra_cfg.runtime.choices['lifting_functions@pykoop_pipeline'])
    elif cfg.robot == 'soft_robot':
        path = "build/pykoop_objects/{}/variance_{}/kp_{}_{}.bin".format(
            cfg.robot, cfg.variance,
            hydra_cfg.runtime.choices['regressors@pykoop_pipeline'],
            hydra_cfg.runtime.choices['lifting_functions@pykoop_pipeline'])
    elif cfg.robot == 'msd':
        path = "build/pykoop_objects/{}/variance_{}/kp_{}_{}.bin".format(
            cfg.robot, cfg.variance,
            hydra_cfg.runtime.choices['regressors@pykoop_pipeline'],
            hydra_cfg.runtime.choices['lifting_functions@pykoop_pipeline'])
    os.makedirs(os.path.dirname(path), exist_ok=True)

    hydra_cfg = HydraConfig.get()

    # Get preprocessed data
    with open(
            "build/preprocessed_data/{}/variance_{}.bin".format(
                cfg.robot, cfg.variance), "rb") as f:
        data = pickle.load(f)

    regressor = hydra_cfg.runtime.choices['regressors@pykoop_pipeline']

    # Save the episode feature
    path2 = "build/others/"
    os.makedirs(os.path.dirname(path2), exist_ok=True)
    with open(path2 + "ep_feat.bin", "wb") as f:
        pickle.dump(data.pykoop_dict['X_train'][:, [0, 1]].T, f)

    with open(path2 + "variance.bin", "wb") as f:
        pickle.dump(cfg.variance, f)

    with open(path2 + "robot.bin", "wb") as f:
        pickle.dump(cfg.robot, f)

    # Resets the n_sv file (by deleting it)
    if os.path.exists("build/others/n_sv_to_keep.bin"):
        os.remove("build/others/n_sv_to_keep.bin")

    # Train model
    kp.fit(data.pykoop_dict['X_train'],
           n_inputs=data.pykoop_dict['n_inputs'],
           episode_feature=True)

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)

    if regressor == 'TEDMD-AS' or regressor == 'TEDMD':
        with open("build/others/variance.bin", "wb") as f:
            pickle.dump(0, f)

        with open(
                "build/preprocessed_data/{}/variance_{}.bin".format(
                    cfg.robot, 0), "rb") as f:
            data_true = pickle.load(f)

        kp_true.fit(data_true.pykoop_dict['X_train'],
                    n_inputs=data_true.pykoop_dict['n_inputs'],
                    episode_feature=True)

    # Calculate the Frobenius error here
    path3 = "build/pykoop_objects/soft_robot/true_kp/"
    os.makedirs(os.path.dirname(path3), exist_ok=True)
    if regressor == 'TEDMD-AS' or regressor == 'TEDMD':
        with open(path3 + "var_{}.bin".format(cfg.variance), "wb") as f:
            pickle.dump(kp_true, f)
    # print(scipy.linalg.eig(kp.regressor_.coef_.T[:15, :15])[0])
    # print(kp.regressor_.coef_.T.shape)

    if cfg.pred:
        if regressor == 'TEDMD-AS' or regressor == 'EDMD-AS' or regressor == 'FBEDMD-AS':
            if cfg.variance <= 1:
                kp.x_pred = kp.predict_trajectory(
                    data.pykoop_dict['x0_valid'],
                    data.pykoop_dict['u_valid'],
                    relift_state=True,
                    return_lifted=False,
                )

    with open(path, "wb") as f:
        data_dump = pickle.dump(kp, f)

    test = 1


if __name__ == '__main__':
    main()
