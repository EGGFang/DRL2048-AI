import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Actor_Critic(nn.Module):
    def __init__(self, input, hidden, output):
        super(Actor_Critic, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output),
            nn.Softmax(),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.sequential1(x)
        actor = self.actor(x)  # Q value of action space
        critic = self.critic(x)  # predict reward

        return actor, critic


class A2C():
    def __init__(self, input, hidden, output, gamma=0.9, priority=True):
        self.actor_critic = Actor_Critic(input, hidden, output).to(device)
        self.opt = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.priority = priority

        self.gamma = gamma

    def get_action(self, state):
        state = state.to(device).float()
        action_prob, _ = self.actor_critic(state)

        return action_prob

    def update(self, replay):
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

        q_actor, q_critic = self.actor_critic(state1_batch)
        dist = torch.distributions.Categorical(q_actor)
        log_probs = dist.log_prob(action_batch.squeeze(-1)).unsqueeze(-1)

        _, q_next = self.actor_critic(state2_batch)
        td_target = reward_batch+self.gamma*q_next.detach()
        advantages = (q_critic-td_target).detach()

        if(self.priority == True):
            loss_critic = (advantages**2*weight).mean()
        else:
            loss_critic = (advantages**2).mean()

        loss_actor = (-log_probs*advantages.detach()).mean()

        entropy = dist.entropy().mean()
        entropy_bouns = -0.001*entropy

        loss = loss_actor+0.5*loss_critic+entropy_bouns

        if(self.priority == True):
            replay.update(tree_idx, loss_critic.reshape(-1).detach().to("cpu"))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss_actor.item(), loss_critic.item(), loss.item()

    def save_model(self, path):
        torch.save(self.actor_critic, path)

    def load_model(self, path):
        model = torch.load(path)
        return model
