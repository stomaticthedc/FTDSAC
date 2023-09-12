import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import QNetwork


class DDQN(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.Q = QNetwork(num_inputs, action_space, args.hidden_size).to(device=self.device)
        self.Q_target = QNetwork(num_inputs, action_space, args.hidden_size).to(device=self.device)
        self.Q_optim = Adam(self.Q.parameters(), lr=args.lr)

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

    def select_action(self, state, eps=0.1):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if np.random.rand() > eps:
            with torch.no_grad():
                return self.Q(state).max(1)[1].data[0].item()
        else:
            return np.random.randint(0, self.Q.num_outputs)

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(np.float32(done_batch)).to(self.device).unsqueeze(1)

        current_Q_values = self.Q(state_batch).gather(1, action_batch)
        max_next_Q_values = self.Q_target(next_state_batch).max(1)[0].unsqueeze(1)
        expected_Q_values = reward_batch + self.gamma * max_next_Q_values * (1 - done_batch)

        loss = F.mse_loss(current_Q_values, expected_Q_values.detach())

        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()

        if updates % self.target_update_interval == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return loss.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", Q_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if Q_path is None:
            Q_path = "models/ddqn_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(Q_path))
        torch.save(self.Q.state_dict(), Q_path)

    # Load model parameters
    def load_model(self, Q_path):
        print('Loading models from {}'.format(Q_path))
        if Q_path is not None:
            self.Q.load_state_dict(torch.load(Q_path))
