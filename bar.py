import numpy as np

a = np.load("/workspace/callback_results/rl_sobc_maze_seed1/evaluations.npz")

# for k, v in a.items():
#     print(k)

# print(a["results"].shape)
for val in a["results"][:100][::2]:
    success = (val >= 1.0)
    print(np.mean(success))
