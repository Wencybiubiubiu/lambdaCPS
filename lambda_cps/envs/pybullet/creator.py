import pickle
from typing import Tuple

import networkx as nx
import numpy as np
import pybullet as p
import pybullet_data

from lambda_cps.config import DATA_ROOT


def get_attr(obj, name, default_value=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    return default_value


class DesignCreator:
    def __init__(self,
                 robot_type: str):
        supported_types = ("2d-arm",)
        if robot_type not in supported_types:
            raise NotImplementedError()
        self.robot_type = robot_type

    @staticmethod
    def _build_world(gui: bool) -> Tuple[int, int]:
        client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        return client_id, plane_id

    def from_networkx(self,
                      nx_graph: nx.MultiDiGraph,
                      gui: bool = True) -> int:
        client_id, plane_id = self._build_world(gui)
        if self.robot_type == "2d-arm":
            self.build_2d_arm(nx_graph, client_id, plane_id)

        return client_id

    def build_2d_arm(self,
                     nx_graph: nx.MultiDiGraph,
                     client_id: int,
                     plane_id: int):
        # link <-> node, joint <-> edge
        # find the world link
        world_link: nx.Node = None
        for node in nx_graph.nodes:
            world_link = node
            if nx_graph.nodes[node]["label"] == "environment":
                break
        assert nx_graph.nodes[world_link]["label"] == "environment"

        link_order = list(nx.traversal.dfs_preorder_nodes(nx_graph, world_link))
        joint_order = list(nx.traversal.dfs_edges(nx_graph, world_link))
        assert len(link_order) == len(joint_order) + 1

        # create base
        base = nx_graph.nodes[world_link]
        base_radius = get_attr(base, "radius", 0.1)
        base_mass = get_attr(base, "mass", 1)
        base_pos = get_attr(base, "pos", [0, 0, 0.1])
        base_color = get_attr(base, "color", [0, 1, 0, 1])

        base_vid = p.createVisualShape(p.GEOM_SPHERE,
                                       rgbaColor=base_color,
                                       radius=base_radius,
                                       physicsClientId=client_id)
        base_cid = p.createCollisionShape(p.GEOM_SPHERE,
                                          radius=base_radius,
                                          physicsClientId=client_id)
        base_id = p.createMultiBody(base_mass, base_cid, base_vid, base_pos,
                                    physicsClientId=client_id)
        # pin base to the ground
        p.createConstraint(plane_id, -1, base_id, -1, p.JOINT_FIXED,
                           [0, 0, 0], [0, 0, 0], [0, 0, 0])

        # create arm
        prev_link = base_id
        link_radius = get_attr(nx_graph.nodes[link_order[1]], "radius", 0.1)
        link_pos = [0, base_radius - link_radius, base_radius]
        connected_base = False

        for link, joint in zip(link_order[1:], joint_order):
            # get link attributes
            link_obj = nx_graph.nodes[link]
            link_mass = get_attr(link_obj, "mass", 1.0)
            link_radius = get_attr(link_obj, "radius", 0.1)
            link_length = get_attr(link_obj, "length", 1.0)
            link_color = get_attr(link_obj, "color", [0, 0.5, 0.8, 1])

            # create link object
            link_vid = p.createVisualShape(p.GEOM_CAPSULE,
                                           rgbaColor=link_color,
                                           radius=link_radius,
                                           length=link_length,
                                           physicsClientId=client_id)
            link_cid = p.createCollisionShape(p.GEOM_CAPSULE,
                                              radius=link_radius,
                                              height=link_length,
                                              physicsClientId=client_id)

            link_pos[1] += link_length / 2 + link_radius * 2
            link_id = p.createMultiBody(
                link_mass,
                link_cid,
                link_vid,
                link_pos,
                baseOrientation=p.getQuaternionFromEuler([-np.pi / 2, 0, 0]),
                physicsClientId=client_id
            )
            link_pos[1] += link_length / 2

            # get joint attribute
            # joint_obj = nx_graph.edges[joint]

            # create joint with constraint
            # TODO: replace constraints with
            # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/urdfEditor.py
            if not connected_base:
                parent_frame_loc = [0, 0, 0]
                child_frame_loc = [0, 0, -link_length / 2 - link_radius - 0.1]
                p.createConstraint(prev_link, -1, link_id, -1, p.JOINT_POINT2POINT,
                                   [0, 0, 0], parent_frame_loc, child_frame_loc)
                connected_base = True
            else:
                parent_frame_loc = [0, 0, link_length / 2 + link_radius]
                child_frame_loc = [0, 0, -link_length / 2 - link_radius]
                p.createConstraint(prev_link, -1, link_id, -1, p.JOINT_POINT2POINT,
                                   [0, 0, 0], parent_frame_loc, child_frame_loc)
            prev_link = link_id


if __name__ == '__main__':
    creator = DesignCreator("2d-arm")
    with open(f"{DATA_ROOT}/test/design_nx.pkl", "rb") as f:
        g = pickle.load(f)
    creator.from_networkx(g)
    while True:
        p.stepSimulation()
