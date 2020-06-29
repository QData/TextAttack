import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax

import textattack


class AdaptiveSoftmax(nn.Module):
    def __init__(self, input_size, cutoffs, scale_down=4):
        super().__init__()
        self.input_size = input_size
        self.cutoffs = cutoffs
        self.output_size = cutoffs[0] + len(cutoffs) - 1
        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        for i in range(len(cutoffs) - 1):
            seq = nn.Sequential(
                nn.Linear(input_size, input_size // scale_down, False),
                nn.Linear(input_size // scale_down, cutoffs[i + 1] - cutoffs[i], False),
            )
            self.tail.append(seq)

    def reset(self, init=0.1):
        self.head.weight.data.uniform_(-init, init)
        for tail in self.tail:
            for layer in tail:
                layer.weight.data.uniform_(-init, init)

    def set_target(self, target):
        self.id = []
        for i in range(len(self.cutoffs) - 1):
            mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))
            if mask.sum() > 0:
                self.id.append(Variable(mask.float().nonzero().squeeze(1)))
            else:
                self.id.append(None)

    def forward(self, inp):
        assert len(inp.size()) == 2
        output = [self.head(inp)]
        for i in range(len(self.id)):
            if self.id[i] is not None:
                output.append(self.tail[i](inp.index_select(0, self.id[i])))
            else:
                output.append(None)
        return output

    def log_prob(self, inp):
        assert len(inp.size()) == 2
        head_out = self.head(inp)
        n = inp.size(0)
        prob = torch.zeros(n, self.cutoffs[-1]).to(textattack.shared.utils.device)
        lsm_head = log_softmax(head_out, dim=head_out.dim() - 1)
        prob.narrow(1, 0, self.output_size).add_(
            lsm_head.narrow(1, 0, self.output_size).data
        )
        for i in range(len(self.tail)):
            pos = self.cutoffs[i]
            i_size = self.cutoffs[i + 1] - pos
            buff = lsm_head.narrow(1, self.cutoffs[0] + i, 1)
            buff = buff.expand(n, i_size)
            temp = self.tail[i](inp)
            lsm_tail = log_softmax(temp, dim=temp.dim() - 1)
            prob.narrow(1, pos, i_size).copy_(buff.data).add_(lsm_tail.data)
        return prob


class AdaptiveLoss(nn.Module):
    def __init__(self, cutoffs):
        super().__init__()
        self.cutoffs = cutoffs
        self.criterions = nn.ModuleList()
        for i in self.cutoffs:
            self.criterions.append(nn.CrossEntropyLoss(size_average=False))

    def reset(self):
        for criterion in self.criterions:
            criterion.zero_grad()

    def remap_target(self, target):
        new_target = [target.clone()]
        for i in range(len(self.cutoffs) - 1):
            mask = target.ge(self.cutoffs[i]).mul(target.lt(self.cutoffs[i + 1]))

            if mask.sum() > 0:
                new_target[0][mask] = self.cutoffs[0] + i
                new_target.append(target[mask].add(-self.cutoffs[i]))
            else:
                new_target.append(None)
        return new_target

    def forward(self, inp, target):
        n = inp[0].size(0)
        target = self.remap_target(target.data)
        loss = 0
        for i in range(len(inp)):
            if inp[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= inp[i].size(1)
                criterion = self.criterions[i]
                loss += criterion(inp[i], Variable(target[i]))
        loss /= n
        return loss
