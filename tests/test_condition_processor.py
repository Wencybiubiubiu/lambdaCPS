
from lambda_cps.evaluation.reward.condition import ConditionProcessor
import numpy as np
from lambda_cps.evaluation.simulation_sampling.sampler import Sampler


def test_condition_processor(num_of_sampling_in_each_design):
    pre_condition_high = np.array([-np.pi / 2, 0.1])
    pre_condition_low = -pre_condition_high
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition = pre_condition

    output_sample_vector = []

    cond = ConditionProcessor(pre_condition, post_condition)
    for i in range(num_of_sampling_in_each_design):
        valid_new_state = cond.initial_state_generator()

        # evaluate process
        # There should be a evaluation function that return the result state
        # To test functionality of condition processor, we temporarily use random pre_condition to check

        max_diff = cond.score_calculator(valid_new_state)

        output_sample_vector += [[valid_new_state, max_diff]]

    for i in output_sample_vector:
        print(i)
    return output_sample_vector


test_condition_processor(10)

