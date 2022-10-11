from typing import Callable, overload
import torch
import torch.nn as nn
from . import surrogate
from .neuron_spikingjelly import IFNode, LIFNode

class OnlineIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = None,
            surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = True,
            track_rate: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike


class OnlineLIFNode(LIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
            v_reset: float = None, surrogate_function: Callable = surrogate.Sigmoid(),
            detach_reset: bool = True, track_rate: bool = True, neuron_dropout: float = 0.0, **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset)
        self.track_rate = track_rate
        self.dropout = neuron_dropout
        if self.track_rate:
            self.register_memory('rate_tracking', None)
        if self.dropout > 0.0:
            self.register_memory('mask', None)

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau

        if self.v_reset is None or self.v_reset == 0:
            self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    # should be initialized at the first time step
    def forward_init(self, x: torch.Tensor):
        self.v = torch.zeros_like(x)
        self.rate_tracking = None
        if self.dropout > 0.0 and self.training:
            self.mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            self.mask = self.mask.requires_grad_(False) / (1 - self.dropout)

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        output_type = kwargs.get('output_type', 'spike')
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if self.dropout > 0.0 and self.training:
            spike = self.mask.expand_as(spike) * spike

        if save_spike:
            self.spike = spike

        if self.track_rate:
            with torch.no_grad():
                if self.rate_tracking == None:
                    self.rate_tracking = spike.clone().detach()
                else:
                    self.rate_tracking = self.rate_tracking * (1 - 1. / self.tau) + spike.clone().detach()

        if output_type == 'spike_rate':
            assert self.track_rate == True
            return torch.cat((spike, self.rate_tracking), dim=0)
        else:
            return spike
