import argparse
import gym, d4rl

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='door', choices=['door', 'hammer', 'relocate', 'pen'])
parser.add_argument('--num_eps', type=int, default=50)

args = parser.parse_args()

env = gym.make(f'{args.env}-human-v0')
done = False

for e in range(args.num_eps):
    print(f"Starting ep {e} of {args.num_eps}")
    obs = env.reset()
    env.render()
    while not done:
        next_obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
    done = False
