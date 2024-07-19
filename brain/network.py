import torch
import torch.nn as nn
import torch.nn.functional as F

class CElegansBrain(nn.Module):
    def __init__(self):
        super(CElegansBrain, self).__init__()

        # these are from common literature
        self.num_neurons = 302
        self.num_sensory_neurons = 60
        self.num_motor_neurons = 94
        self.num_interneurons = 148
        self.num_light_sensors = 30
        self.num_vibration_sensors = 30

        # Change neuron_states from Parameter to regular tensor
        self.register_buffer('neuron_states', torch.zeros(self.num_neurons))

        self.chemical_synapses = nn.Parameter(torch.randn(self.num_neurons, self.num_neurons) * 0.01)
        self.gap_junctions = nn.Parameter(torch.randn(self.num_neurons, self.num_neurons) * 0.01)
        self.mask = torch.ones(self.num_neurons, self.num_neurons) - torch.eye(self.num_neurons)
        self.tau = nn.Parameter(torch.ones(self.num_neurons) * 0.1)

        self.light_scale = nn.Parameter(torch.ones(self.num_light_sensors) * 0.1)
        self.vibration_scale = nn.Parameter(torch.ones(self.num_vibration_sensors) * 0.1)

    def forward(self, light_input, vibration_input, dt=0.1):
        light_input = torch.tensor(light_input, dtype=torch.float32)
        vibration_input = torch.tensor(vibration_input, dtype=torch.float32)

        # copy of neuron_states for this forward pass
        states = self.neuron_states.clone()

        # apply sensory inputsin-place
        sensory_input = torch.zeros_like(states)
        sensory_input[:self.num_light_sensors] = light_input * self.light_scale
        sensory_input[self.num_light_sensors:self.num_sensory_neurons] = vibration_input * self.vibration_scale
        states = states + sensory_input

        # obv not actually how a worm brain works... need to improve?
        chemical_input = F.relu(torch.matmul(states, self.chemical_synapses * self.mask))
        gap_input = torch.matmul(states, self.gap_junctions * self.mask)
        d_states = (-states + chemical_input + gap_input) / self.tau
        states = states + d_states * dt
        states = F.relu(states)

        # update neuron_states
        self.neuron_states = states.detach()

        return states[-self.num_motor_neurons:]

    def reset(self):
        self.neuron_states.zero_()
