import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lambda_cps.config import exp_res_path
from lambda_cps.envs import Pendulum
from lambda_cps.evaluation.simulation import rs_simulation


def eval_param(param_name: str, param_range):
    env = Pendulum()
    res = {
        param_name: [],
        "rewards": []
    }
    for param in param_range:
        env.set_param(param_name, param)
        print(f"{param_name}={param}: ")
        rewards = rs_simulation(env, 200, 10, mpc_traj_len=10, n_sample=100)
        res[param_name].extend([param] * len(rewards))
        res["rewards"].extend(rewards)
    res_df = pd.DataFrame(res)

    res_path = exp_res_path(exp_exe_path=__file__) + f"/{param_name}.pdf"
    sns.lineplot(data=res_df, x=param_name, y="rewards")
    plt.savefig(res_path)
    plt.clf()

    return res_df


if __name__ == '__main__':
    param_names_to_range = {"g": np.arange(1, 10, 2),
                            "m": np.arange(0.1, 3, 0.5),
                            "l": np.arange(0.1, 3, 0.5)}

    for pn in param_names_to_range:
        eval_param(pn, param_names_to_range[pn])
