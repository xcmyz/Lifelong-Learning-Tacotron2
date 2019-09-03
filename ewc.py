import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from utils import to_gpu


class EWC(nn.Module):
    """ Elastic Weight Consolidation """

    def __init__(self, model, dataset):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.params = dict()

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params.update({n: p})

        self._means = dict()
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means.update({n: to_gpu(p.data)})

    def _diag_fisher(self):
        precision_matrices = dict()
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices.update({n: to_gpu(p.data)})

        self.model.train()
        cnt = 0
        for batch in self.dataset:
            cnt += 1
            if cnt % 20 == 0:
                print("Fisher Information Done", cnt)

            self.model.zero_grad()
            inputs, target = self.model.module.parse_batch(batch)
            mel, mel_postnet, gate, _ = self.model.forward(inputs)

            mel_loss = nn.MSELoss()(mel, target[0])
            mel_postnet_loss = nn.MSELoss()(mel_postnet, target[0])
            gate_loss = nn.BCEWithLogitsLoss()(gate, target[1])

            loss = mel_loss + mel_postnet_loss + gate_loss
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data = precision_matrices[n].data + \
                    (p.grad.data**2) / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            temp_loss = self._precision_matrices[n] * (p - self._means[n])**2
            loss += temp_loss.sum()
        return loss
