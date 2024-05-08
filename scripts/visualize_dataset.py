import argparse
import d4rl
import gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v0')
    parser.add_argument('--use_acts', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_name)

    dataset = env.get_dataset()
    if 'infos/qpos' not in dataset:
        raise ValueError('Only MuJoCo-based environments can be visualized')
    qpos = dataset['infos/qpos']
    qvel = dataset['infos/qvel']
    rewards = dataset['rewards']
    actions = dataset['actions']
    dones = dataset['terminals']

    env.reset()
    env.set_state(qpos[0], qvel[0])
    for t in range(qpos.shape[0]):
        if args.use_acts:
            obs, rew, done, info = env.step(actions[t])
            if dones[t]:
                env.reset()
                env.set_state(qpos[t+1], qvel[t+1])
        else:
            env.set_state(qpos[t], qvel[t])
        env.render()
