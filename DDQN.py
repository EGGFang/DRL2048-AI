import torch.nn as nn
import torch.nn.functional as F
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class QNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(QNet, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.Linear(input, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )
        self.sequential2 = nn.Sequential(
            nn.Conv2d(18, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.Flatten(1),
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

        self.sequential3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output),
        )

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        # x = self.sequential1(x)

        x = x.view(x.shape[0], 1, 4, 4)
        x = self.sequential3(x)
        return x


class DDQN():
    def __init__(self, input, hidden, output, gamma=0.99, target_update_step=3000, priority=True):
        self.gamma = gamma
        self.target_update_step = target_update_step
        self.policy_model = QNet(input, hidden, output).to(device)
        self.target_model = QNet(input, hidden, output).to(device)

        self.opt = torch.optim.Adam(self.policy_model.parameters(), lr=0.0001)
        self.priority = priority

        self.steps = 0

    def get_action(self, state):
        self.policy_model.eval()
        action = self.policy_model(state.float().to(device))

        return action

    def soft_update(self, tau=0.001):
        for target_params, policy_params in zip(self.target_model.parameters(), self.policy_model.parameters()):
            target_params.data.copy_(tau * policy_params + (1 - tau) * target_params)

    def update(self, replay):
        self.steps += 1
        self.policy_model.train()

        if(self.priority == True):
            batch_memory, (tree_idx, weight) = replay.sample()
            state1_batch = torch.FloatTensor(batch_memory[:, :16]).reshape(replay.batch_size, 1, 4, 4).to(device)
            action_batch = torch.LongTensor(batch_memory[:, 16: 16+1].astype(int)
                                            ).reshape(replay.batch_size, 1).to(device)
            reward_batch = torch.FloatTensor(batch_memory[:, 16+1: 16+2]).reshape(replay.batch_size, 1).to(device)
            state2_batch = torch.FloatTensor(batch_memory[:, -16:]).reshape(replay.batch_size, 1, 4, 4).to(device)
            weight = torch.FloatTensor(weight).to(device)
        else:
            state1_batch, action_batch, reward_batch, state2_batch, done_batch = replay.sample()
            state1_batch = state1_batch.reshape(replay.batch_size, 1, 4, 4)
            state2_batch = state2_batch.reshape(replay.batch_size, 1, 4, 4)

        q_value_batch = self.policy_model(state1_batch.float()).gather(
            dim=1, index=action_batch.long())

        next_q_batch = self.policy_model(state2_batch.float())
        next_target_q_batch = self.target_model(state2_batch.float())
        next_max_q_value_batch = next_target_q_batch.gather(1, torch.max(next_q_batch, 1)[1].unsqueeze(1))

        expected_q_value_batch = reward_batch+self.gamma*next_max_q_value_batch

        td_error = torch.abs(q_value_batch-expected_q_value_batch)
        if(self.priority == True):
            loss = (td_error**2*weight).mean()
        else:
            loss = (td_error**2).mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_model.parameters(), 1.0)
        self.opt.step()

        if(self.steps % 200 == 0):
            self.soft_update(0.1)

        if(self.priority == True):
            replay.update(tree_idx, td_error.reshape(-1).detach().to("cpu"))

        return loss.item()

    def save_model(self, path):
        torch.save(self.policy_model, path)

    def load_model(self, path):
        model = torch.load(path)
        return model
