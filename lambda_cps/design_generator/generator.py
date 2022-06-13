import random

class DesignGenerator:

    def __init__(self, production_rules):
        self.production_rules = production_rules
        return

    # The functions outside the pipeline class should be elaborated in the future (wrapped in class)
    def get_a_new_design(self, learned_model, init_sketch, init_xml_file):
        mass = random.uniform(0, 5)
        length = random.uniform(1, 10)

        connection_mat = [[0, 1], [1, 0]]
        feature_mat = [[0, 0], [mass, length]]

        design_matrix = [connection_mat, feature_mat]
        new_xml_file = init_xml_file  # It should be updated instead of using initial one
        ancestors = None  # It should be all designs in the generating process of the target complete design
        return design_matrix, new_xml_file, ancestors
