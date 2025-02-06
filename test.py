import numpy as np
import torch

import Env
import warnings

warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)


def state_process(state):
    state = np.reshape(state, -1)
    state = np.log2(state+1)/16
    new_state = state.reshape(1, 4, 4)

    return new_state


game_time = 10000
model = torch.load("./model_99000.pt").to("cpu")
env = Env.Game2048Env()

reward_total = 0
highest_count = np.zeros(12)

for i in range(game_time):
    state1, _, _, _ = env.reset()
    state1 = state_process(state1)
    state1 = torch.from_numpy(state1)

    done = False
    while(done == False):
        qvalue = model(state1.reshape(1, 1, 4, 4).float())
        action = torch.argmax(qvalue)
        state2, reward, done, info = env.step(action)
        # env.draw_board(state2)
        state2 = state_process(state2)
        state2 = torch.from_numpy(state2)
        state1 = state2
        reward_total += reward

    highest_count[:int(np.log2(env.highest()))] += 1


highest_count /= game_time
reward_avg = reward_total / game_time

print(highest_count)
print(reward_avg)
