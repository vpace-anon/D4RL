import click
import argparse
import os
import gym
import numpy as np
import pickle
import d4rl
from mjrl.utils.gym_env import GymEnv
DESC = '''
Helper script to visualize demonstrations.\n
USAGE:\n
    Visualizes demonstrations on the env\n
    $ python utils/visualize_demos --env_name relocate-v0\n
'''


def main(env_name):
    if env_name is "":
        print("Unknown env.")
        return

    env = gym.make(env_name)
    dataset = env.get_dataset()

    ds_str = f"{env_name.split('-human-v0')[0]}-v0_demos.pickle"
    raw_demos_file = os.path.join(os.environ['VPACE_TOP_DIR'], "raw_hand_dapg_data", ds_str)
    raw_dataset = pickle.load(open(raw_demos_file, 'rb'))

    for path in raw_dataset:
        env.reset()
        env.set_env_state(path['init_state_dict'])
        actions = path['actions']
        for t in range(actions.shape[0]):
            obs, reward, done, info = env.step(actions[t])
            #print(f"Palm to Handle Dist: {np.linalg.norm(obs[-4:-1])}")
            #print(f"Obs last 13 to end: {obs[-13:]}")
            # print(info['aux_dones'])
            env.render()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door-human-v0')
    args = parser.parse_args()

    main(args.env_name)