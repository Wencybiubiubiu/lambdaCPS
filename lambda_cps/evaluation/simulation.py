import copy

from tqdm import trange

from lambda_cps.evaluation.control.random_shooting import RandomShootingController


def rs_simulation(env, n_step, n_traj, mpc_traj_len=20, n_sample=10, render=False):
    env_model = copy.deepcopy(env)
    rs_controller = RandomShootingController(env_model)

    all_rewards = []
    for _ in trange(n_traj):
        env.reset()
        rew_sum = 0
        for _ in range(n_step):
            env_model.set_state(env.get_state())
            action = rs_controller.next_actions(mpc_traj_len, n_sample)[0]
            _, r, _, _ = env.step(action)
            rew_sum += r
            if render:
                env.render()
        all_rewards.append(rew_sum)

    return all_rewards
