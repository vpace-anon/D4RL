import importlib
import pickle

import d4rl.hand_manipulation_suite.raw_human_demonstrations as demos


data = pickle.load(importlib.resources.files(demos).joinpath('door-v0_demos.pickle').open('rb'))

print(f"Num trajs: {len(data)}\n ex obs: \n{data[0]['observations'][0]}")