import time

import numpy as np
import pybullet as p

from lambda_cps.envs.pybullet.reacher import Reacher, ReacherController


def get_all_joints_info():
    robot = Reacher(gui=False)
    print(robot.robot_id)

    n_joints = p.getNumJoints(bodyUniqueId=robot.robot_id)

    for i in range(n_joints):
        print(p.getJointInfo(bodyUniqueId=robot.robot_id, jointIndex=i))


def get_all_link_info():
    robot = Reacher(gui=False)
    n_joints = p.getNumJoints(bodyUniqueId=robot.robot_id)

    for i in range(n_joints):
        print(p.getLinkState(robot.robot_id, linkIndex=i))


def test_ctrl():
    controller = ReacherController(Reacher())
    goals = [
        np.array([0.1, -0.1]),
        np.array([0.12, 0.1]),
        np.array([-0.1, -0.12]),
        np.array([0.1, -0.1]),
        np.array([0.12, 0.1]),
        np.array([-0.1, -0.12])
    ]

    for goal in goals:
        for _ in range(240 // 4):
            controller.ctrl(goal)
            p.stepSimulation()
            time.sleep(0.1)


if __name__ == '__main__':
    # get_all_joints_info()
    # get_all_link_info()
    test_ctrl()
