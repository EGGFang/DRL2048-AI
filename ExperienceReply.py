import random
import torch
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ExperienceReply:  # 經驗池
    def __init__(self, N=500, batch_size=100):
        self.N = N  # 經驗池容量
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0

    def add(self, state1, action, reward, state2, done):
        self.counter += 1
        if(self.counter % 500 == 0):  # 經驗池內容洗牌，增加隨機性
            self.shuffle_memory()

        if(len(self.memory) < self.N):  # 判斷經驗池容量是否達上限
            self.memory.append((state1, action, reward, state2, done))  # 未達上限->增加
        else:
            rand_index = np.random.randint(0, self.N-1)
            self.memory[rand_index] = (state1, action, reward, state2, done)  # 達上限->替換

    def shuffle_memory(self):  # 經驗池洗牌
        random.shuffle(self.memory)

    def sample(self):
        batch_size = min(len(self.memory), self.batch_size)

        if(len(self.memory) < 1):
            print("No data in memory")
            return None

        ind = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)  # 從經驗池中隨機選擇訓練資料
        batch = [self.memory[i] for i in ind]
        state1_batch = torch.stack([x[0].squeeze(dim=0) for x in batch], dim=0).to(device)
        action_batch = torch.Tensor([x[1] for x in batch]).to(device).reshape(batch_size, 1)
        reward_batch = torch.Tensor([x[2] for x in batch]).to(device).reshape(batch_size, 1)
        state2_batch = torch.stack([x[3].squeeze(dim=0) for x in batch], dim=0).to(device)
        done_batch = torch.Tensor([x[4] for x in batch]).to(device).reshape(batch_size, 1)

        return state1_batch, action_batch, reward_batch, state2_batch, done_batch
