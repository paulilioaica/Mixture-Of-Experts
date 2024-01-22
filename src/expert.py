import torch.nn as nn


class FeedForwardExpert(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.lienar3 =  nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear3(nn.functional.silu(self.linear1(x)) * self.linear2(x))