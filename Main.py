import DDQN
import D3QN
import A2C
import ExperienceReply
import PrioritizedReplay

import torch

import numpy as np
import matplotlib.pyplot as plt
import Env
import warnings
import os

warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

params = {
    "batch_size": 256,
    "gamma": 0.9,  # 削減因子
    "epochs": 300000,
    "gamma": 0.9,
    "per_epoch_train": 1,
    "check_point": True,
    "priority": True,
}

env = Env.Game2048Env()
if(params["priority"] == True):
    replay = PrioritizedReplay.PrioritizedReplay(10000, 128, 1*4*4)
else:
    replay = ExperienceReply.ExperienceReply(100000, 128)
# model = D3QN.D3QN(1*4*4, 64, 4)
model = DDQN.DDQN(1*4*4, 64, 4)
# model = A2C.A2C(16, 64, 4)

init_eps = 0.15
eps = init_eps
epoch = 0
best_score = 0

loss_history = []
reward_history = []
score_history = []
highest_history = []

reward_list = []
score_list = []

if(params["check_point"] and len(os.listdir("./check_point")) > 0):
    check_point = np.load("./check_point/check_point.npy", allow_pickle="TRUE").item()

    loss_history = check_point["loss_history"]
    reward_history = check_point["reward_history"]
    score_history = check_point["score_history"]
    highest_history = check_point["highest_history"]
    reward_list = check_point["reward_list"]
    score_list = check_point["score_list"]
    epoch = check_point["epoch"]
    eps = check_point["eps"]
    best_score = check_point["best_score"]
    replay = check_point["replay"]
    model.policy_model = check_point["policy_model"]
    model.target_model = check_point["target_model"]
    model.opt = check_point["opt"]


def test(render, turns=3):
    scores = 0
    turn = 0
    test_env = Env.Game2048Env()
    highest = 0
    while(turn < turns):
        steps = 0
        done = False

        state1 = test_env.reset()
        state1 = state_process(state1)
        state1 = torch.from_numpy(state1)
        while not done:
            steps += 1
            action = torch.argmax(model.get_action(state1.float().reshape(1, 1, 4, 4).to(device))).to('cpu')
            state2, reward, done = test_env.step(action)
            state2 = state_process(state2)
            state2 = torch.from_numpy(state2)
            state1 = state2

            if render:
                test_env.render()

            if(env.moved == False):
                break

        turn += 1
        scores += test_env.score
        highest = max(env.highest(), highest)

    return scores / turns, highest


def state_process(state):
    # state = np.reshape(state, -1)
    # state[state == 0] = 1
    # state = np.log2(state)
    # state = state.astype(int)
    # state = np.reshape(np.eye(18)[state], -1)
    # new_state = state.reshape(18, 4, 4)

    state = np.reshape(state, -1)
    state = np.log2(state+1)/16
    new_state = state.reshape(1, 4, 4)

    return new_state


def policy(qvalues, eps):
    if(torch.rand(1) < eps):
        index = torch.randint(low=0, high=4, size=(1,))
        return index
    else:
        qvalues = qvalues.reshape(-1)
        action = torch.argmax(qvalues)
        return action


def Do_Training():
    global eps, epoch, best_score
    while(epoch < params["epochs"]):
        epoch += 1
        state1, _, _, _ = env.reset()
        state1 = state_process(state1)
        state1 = torch.from_numpy(state1)

        reward_total = 0

        eps *= 0.998
        if(eps < 0.05):
            eps = init_eps*(1-epoch/params["epochs"])
        for t in range(50000):
            # env.render("rgb_array")

            Q_value = model.get_action(state1.float().reshape(1, 1, 4, 4).to(device)).to('cpu')

            action = int(policy(Q_value, eps))

            state2, reward, done, info = env.step(action)
            reward = np.log2(reward+1)
            state2 = state_process(state2)
            state2 = torch.from_numpy(state2)

            if(params["priority"] == True):
                replay.add(state1.reshape(-1), action, reward, state2.reshape(-1))
            else:
                replay.add(state1.reshape(-1), action, reward, state2.reshape(-1), done)

            reward_total += reward

            state1 = state2

            if(replay.counter >= 10000 and replay.counter % 5 == 0):
                for j in range(params["per_epoch_train"]):
                    loss = model.update(replay)

                    loss_history.append(loss)

            if(done):
                break

        reward_history.append(reward_total)
        score_history.append(env.score)
        highest_history.append(env.highest())

        if(env.score >= best_score):
            best_score = env.score
            model.save_model(f"./result/model/best.pt")
            print(f"Save Best Model In {epoch} Epoch")

        if(epoch % 1000 == 0):
            # score = test(False)
            # score_history.append(score)

            model.save_model(f"./result/model/model_{epoch}.pt")

            if(params["check_point"]):
                check_point = {
                    "loss_history": loss_history,
                    "score_history": score_history,
                    "reward_history": reward_history,
                    "highest_history": highest_history,
                    "reward_list": reward_list,
                    "score_list": score_list,
                    "epoch": epoch,
                    "eps": eps,
                    "best_score": best_score,
                    "replay": replay,
                    "policy_model": model.policy_model,
                    "target_model": model.target_model,
                    "opt": model.opt,
                }

                np.save("./check_point/check_point", check_point)

        if(len(reward_history) > 100):
            reward_list.append(sum(reward_history[-100:])/100)
            score_list.append(sum(score_history[-100:])/100)

            plt.plot(range(len(reward_list)), reward_list)
            plt.savefig("./result/reward.png")
            plt.cla()

            plt.plot(range(len(loss_history)), loss_history)
            plt.savefig("./result/loss.png")
            plt.cla()

            plt.plot(range(len(score_list)), score_list)
            plt.savefig("./result/score.png")
            plt.cla()

            plt.plot(range(len(highest_history)), highest_history)
            plt.savefig("./result/highest.png")
            plt.cla()

        # print(f"Epoch {i} : reward {reward_total}")


Do_Training()
