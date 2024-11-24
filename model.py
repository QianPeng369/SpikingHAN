import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import neuron

def creat_snn_layer(alpha=2.0, tau = 1.0, surrogate="sigmoid", v_threshold=5e-3, snn="PLIF"):
    if snn in ["LIF", "PLIF"]:
        return getattr(neuron, snn)(tau, alpha=alpha, surrogate=surrogate,v_threshold=v_threshold,detach=True)
    elif snn == "IF":
        return neuron.IF(
            alpha=alpha, surrogate=surrogate, v_threshold=v_threshold, detach=True,
        )
    else:
        raise ValueError("Unknown SNN")


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False), # query vector  Qt
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


class SpikingHGNNLayer(nn.Module):

    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, T, alpha, tau, surrogate, neuron, reset, threshold, dropout1, dropout2
    ):
        super(SpikingHGNNLayer, self).__init__()

        self.T = T
        self.alpha = alpha
        self.tau = tau
        self.surrogate = surrogate
        self.neuron = neuron
        self.reset = reset
        self.threshold = threshold

        self.snn_layers = nn.ModuleList()
        self.snn = creat_snn_layer(
            alpha=self.alpha,
            surrogate=self.surrogate,
            v_threshold=self.threshold,
            snn=self.neuron,
        )
        self.shared_gcn_layer = dglnn.GraphConv(in_size, hidden_size , activation=F.elu)
        self.fc = nn.Linear(hidden_size, out_size, bias=False)
        self.semantic_attention = SemanticAttention(in_size = hidden_size)
        self.num_meta_paths = num_meta_paths
        self.droptout1 = dropout1
        self.droptout2 = dropout2

    def forward(self, gs, h):
        semantic_embeddings = []
        h1 = F.dropout(h, p= self.droptout1)
        for i, g in enumerate(gs):
            h2 = self.shared_gcn_layer(g, h1)
            semantic_embeddings.append(h2)
        h3 = self.semantic_attention(torch.stack(semantic_embeddings, dim=1))

        for t in range(self.T):
            if t == 0:
                out_spikes = self.snn(self.fc(h3))
            else:
                out_spikes += self.snn(self.fc(F.dropout(h3, p= self.droptout2)))
        return  out_spikes / self.T


class SpikingHGNN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size, out_size, T, alpha, tau, surrogate, neuron, reset, threshold, dropout1, dropout2
    ):
        super(SpikingHGNN, self).__init__()

        self.layer = SpikingHGNNLayer(num_meta_paths, in_size, hidden_size, out_size, T, alpha, tau, surrogate, neuron, reset, threshold, dropout1, dropout2)


    def forward(self, g, h):
        h = self.layer(g, h)
        return h
