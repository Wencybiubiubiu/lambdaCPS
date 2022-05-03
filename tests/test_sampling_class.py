
from lambda_cps.evaluation.sampling.condition import ConditionProcessor
import numpy as np
from lambda_cps.evaluation.sampling.sampler import Sampler


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

        max_diff = cond.check_post_condition(valid_new_state)

        output_sample_vector += [[valid_new_state, max_diff]]

    for i in output_sample_vector:
        print(i)
    return output_sample_vector


# test_condition_processor(10)


def test_sampler():
    pre_condition_high = np.array([-np.pi / 1, 0.1])
    pre_condition_low = -pre_condition_high
    pre_condition = [pre_condition_low, pre_condition_high]
    post_condition_high = np.array([-np.pi / 2, 0.1])
    post_condition_low = -pre_condition_high
    post_condition = [post_condition_low, post_condition_high]

    num_of_sample_for_each_design = 2
    evaluation_time_for_each_sample = 100
    num_of_designs = 5

    # there should be a function call to get the new design in each round
    # We temporarily set it as 0.1 to test

    test_sampling = Sampler(pre_condition, post_condition, num_of_sample_for_each_design,
                            evaluation_time_for_each_sample, num_of_designs)
    print(test_sampling.sample())


# test_sampler()
