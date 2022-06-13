# evaluator (number_of_samples_of_this_design, current_design, pre-condition theta, post-condition alpha)
#   = [s_1,s_2,s_3,...,s_n], n is the number of samples for current design

from lambda_cps.evaluation.fitting.GNN import ParamName


class Sampler(ParamName):

    def __init__(self, num_of_sample_for_each_design, evaluation_times_for_each_sample):
        super().__init__()
        self.num_of_sample_for_each_design = num_of_sample_for_each_design
        self.evaluation_times_for_each_sample = evaluation_times_for_each_sample

    def evaluate_one_design(self, env, controller, starting_state):

        env.set_state(starting_state)
        obs = starting_state
        rew_sum = 0
        for i in range(self.evaluation_times_for_each_sample):
            action = controller.predict(obs)

            obs, r, _, _ = env.step(action)
            rew_sum += r
            # env.render()

        result_state = env.get_state()

        return result_state

    # Output format:
    # [graph, list_of_samples]
    # graph = [adjacency_matrix, feature_matrix],
    #          adjacency_matrix with size N x N, feature_matrix with size N x d, d is the number of features
    # list_of_samples = [[initial_state, final_state], [initial_state, final_state], ...]
    def sample_one_design(self, env, controller, condition, generated_design):

        output_sample_vector = []

        for i in range(self.num_of_sample_for_each_design):
            valid_new_state = condition.initial_state_generator()
            result_state = self.evaluate_one_design(env, controller, valid_new_state)

            output_sample_vector += [[valid_new_state, result_state]]

        # print(output_sample_vector)
        return {self.CNNT_MATRIX: generated_design[0], self.FEAT_MATRIX: generated_design[1],
                self.TRAJ_VECTOR: output_sample_vector}

