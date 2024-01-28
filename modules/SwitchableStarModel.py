import torch.nn as nn


class SwitchablestarModel(nn.Module):
    def __init__(self, ns, n_intm_layers, challenge_bits):
        super().__init__()
        self.challenge_bits = challenge_bits

        layers = []
        ns_cnt = ns
        for _ in range(n_intm_layers):
            out = int(ns_cnt // 2)
            layers.extend([
                nn.Linear(ns_cnt, out),
                nn.BatchNorm1d(out),
                nn.LeakyReLU()
            ])
            ns_cnt = out
        self.main = nn.Sequential(
            nn.Linear(challenge_bits, ns),
            nn.BatchNorm1d(ns),
            nn.LeakyReLU(),
            *layers,
            nn.Linear(ns_cnt, 4)
        )

    def forward(self, x):
        #x = x.view(-1, self.challenge_bits)
        return self.main(x).squeeze()
