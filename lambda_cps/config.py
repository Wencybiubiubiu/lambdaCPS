import os
from os.path import dirname, abspath
import lambda_cps

DATA_ROOT = dirname(dirname(abspath(lambda_cps.__file__))) + "/data"


def exp_res_path(exp_exe_path: str):
    """experiments results path"""
    res_path = DATA_ROOT + "/res/" + exp_exe_path.split("exps/")[-1][:-3]
    os.makedirs(res_path, exist_ok=True)
    return res_path
