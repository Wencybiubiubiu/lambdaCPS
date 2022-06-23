## Pipeline: 
evaluator (check pre_condition -> controller -> check post_condition) -> training

- Provided info:
    - Pre-condition theta: range of angle
    - Post-condition alpha: range of angle
    - number_of_steps: how many steps a simulator needs
    - number_of_samples_of_this_design: times of sampling of each design
- Functions:
    - control folder: 
        - controller (initial_state, number_of_steps, current_design) = result_state
    - design_generator_folder:
        - design_generate (structure, mass_of_stick) = current_design (structure fixed in pendulum example, mass varied)
    - sampler folder:
        - initial_state_generator (pre-condition) = initial_state (in pendulum example, if pre-condition is theta = [60,90], generator should random(theta))
        - post_condition_check (post-condition, result_state) = | alpha - result_state (angle) |
        - evaluator (number_of_samples_of_this_design, current_design, pre-condition theta, post-condition alpha) = [s_1,s_2,s_3,...,s_n], n is the number of samples for current design
    - fit_model folder:
        - model (mass_of_stick, [s_1,s_2,s_3,...,s_n]) = ( f(mass) -> score ), (mass_of_stick, [s_1,s_2,s_3,...,s_n]) is a single sample
    - a main file to execute the complete pipeline

## Execution

```
lambdaCPS$ python tests/test_sampling_experiment.py
```

- Parameters embedded:
  - num_of_designs 100
  - num_of_sample_for_each_design 20
  - evaluation_time_for_each_sample 100
  - pre_condition = [[np.pi / 2, 0.1], [-np.pi / 2, 0.1]]
  - post_condition = [[np.pi / 4, -np.inf], [-np.pi / 4, np.inf]]
- Model output:
  - Mean squared error: 181.69
  - Loss: 65.62
- Execution time: 101.90922284126282 seconds
- Figures in data/res/sampling_image/MLPR folder
