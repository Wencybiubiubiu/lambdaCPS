import numpy as np
import pybullet as p

from lambda_cps.common.types import Vec2
from lambda_cps.config import ROBOT_MODEL


class Reacher:
    def __init__(self, gui: bool = True):
        self.gui = gui
        self.client_id = None
        self.robot_id = None
        self.movable_joints = None
        self.end_effect = 4

        self._connect()
        self._build_robot()

    def _connect(self):
        if self.gui:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

    def _build_robot(self):
        model_path = f"{ROBOT_MODEL}/reacher.xml"
        self.robot_id = p.loadMJCF(model_path)[-2]

        self.movable_joints = []
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            if p.getJointInfo(self.robot_id, i, physicsClientId=self.client_id)[2] \
                    != p.JOINT_FIXED:
                self.movable_joints.append(i)


class ReacherController:
    def __init__(self, robot: Reacher):
        self.robot = robot

    def ctrl(self, goal: Vec2):
        goal = np.concatenate([goal, [0.01]])
        conf = p.calculateInverseKinematics(self.robot.robot_id,
                                            self.robot.end_effect,
                                            goal,
                                            physicsClientId=self.robot.client_id)

        for joint, pos in zip(self.robot.movable_joints, conf):
            p.setJointMotorControl2(self.robot.robot_id,
                                    joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=pos)
