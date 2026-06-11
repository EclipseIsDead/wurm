import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Bernoulli, Normal

DEFAULT_CONFIG_PATH = Path(__file__).with_name("brain.yaml")


def load_config(path=DEFAULT_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _inverse_softplus(value):
    return math.log(math.exp(value) - 1.0)


class CElegansBrain(nn.Module):
    def __init__(self, config=None, config_path=DEFAULT_CONFIG_PATH):
        super().__init__()

        self.config = config if config is not None else load_config(config_path)
        brain_config = self.config["brain"]
        dynamics_config = brain_config.get("dynamics", {})
        action_config = brain_config.get("action", {})

        self.num_neurons = int(brain_config["num_neurons"])
        self.num_sensory_neurons = int(brain_config["num_sensory_neurons"])
        self.num_motor_neurons = int(brain_config["num_motor_neurons"])
        self.num_interneurons = int(brain_config["num_interneurons"])
        self.num_light_sensors = int(brain_config["num_light_sensors"])
        self.num_vibration_sensors = int(brain_config["num_vibration_sensors"])
        self.dt = float(dynamics_config.get("dt", 0.1))
        self.state_decay = float(dynamics_config.get("state_decay", 1.0))
        self.grip_actions = int(action_config.get("grip_actions", 2))
        self.controls_per_segment = int(action_config.get("controls_per_segment", 2))
        self.continuous_actions = self.num_motor_neurons - self.grip_actions

        # realistic constraint for when config gets manipulated
        expected_sensory = self.num_light_sensors + self.num_vibration_sensors
        if expected_sensory != self.num_sensory_neurons:
            raise ValueError(
                "light + vibration sensors must equal num_sensory_neurons "
                f"({expected_sensory} != {self.num_sensory_neurons})"
            )

        expected_motor = (
            self.grip_actions
            + int(action_config["segments"]) * self.controls_per_segment
        )
        if expected_motor != self.num_motor_neurons:
            raise ValueError(
                "grip actions + segment flex actions must equal num_motor_neurons "
                f"({expected_motor} != {self.num_motor_neurons})"
            )

        self.register_buffer("neuron_states", torch.zeros(self.num_neurons))
        self.register_buffer(
            "mask",
            torch.ones(self.num_neurons, self.num_neurons)
            - torch.eye(self.num_neurons),
        )

        synapse_std = float(dynamics_config.get("chemical_synapse_init_std", 0.01))
        gap_std = float(dynamics_config.get("gap_junction_init_std", 0.01))
        tau_init = float(dynamics_config.get("tau_init", 0.1))
        sensory_scale = float(dynamics_config.get("sensory_scale_init", 0.1))
        action_std = float(action_config.get("continuous_action_std", 0.15))

        self.chemical_synapses = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons) * synapse_std
        )
        self.gap_junctions = nn.Parameter(
            torch.randn(self.num_neurons, self.num_neurons) * gap_std
        )
        self.tau_raw = nn.Parameter(
            torch.full((self.num_neurons,), _inverse_softplus(tau_init))
        )
        self.light_scale = nn.Parameter(
            torch.ones(self.num_light_sensors) * sensory_scale
        )
        self.vibration_scale = nn.Parameter(
            torch.ones(self.num_vibration_sensors) * sensory_scale
        )
        self.action_log_std = nn.Parameter(
            torch.full((self.continuous_actions,), math.log(action_std))
        )
        self.value_head = nn.Linear(self.num_motor_neurons, 1)

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1e-4

    def _sensor_tensor(self, value, expected_size):
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.neuron_states.device,
        ).flatten()
        if tensor.numel() == 1:
            tensor = tensor.repeat(expected_size)
        if tensor.numel() != expected_size:
            raise ValueError(
                f"expected {expected_size} sensor values, got {tensor.numel()}"
            )
        return tensor

    def motor_state(self, light_input, vibration_input, dt=None):
        light_input = self._sensor_tensor(light_input, self.num_light_sensors)
        vibration_input = self._sensor_tensor(
            vibration_input, self.num_vibration_sensors
        )
        dt = self.dt if dt is None else float(dt)

        states = self.neuron_states.clone()
        sensory_input = torch.zeros_like(states)
        sensory_input[: self.num_light_sensors] = light_input * self.light_scale
        sensory_input[self.num_light_sensors : self.num_sensory_neurons] = (
            vibration_input * self.vibration_scale
        )
        states = states + sensory_input

        chemical_input = F.relu(states @ (self.chemical_synapses * self.mask))
        gap_input = states @ (self.gap_junctions * self.mask)
        d_states = (-states * self.state_decay + chemical_input + gap_input) / self.tau
        states = F.relu(states + d_states * dt)

        with torch.no_grad():
            self.neuron_states.copy_(states.detach())

        return states[-self.num_motor_neurons :]

    def forward(self, light_input, vibration_input, dt=None):
        motor_state = self.motor_state(light_input, vibration_input, dt)
        grip_prob = torch.sigmoid(motor_state[: self.grip_actions])
        flex = torch.sigmoid(motor_state[self.grip_actions :])
        return torch.cat([grip_prob, flex])

    def policy(self, light_input, vibration_input, dt=None):
        motor_state = self.motor_state(light_input, vibration_input, dt)
        grip_logits = motor_state[: self.grip_actions]
        continuous_mean = torch.sigmoid(motor_state[self.grip_actions :])
        value = self.value_head(motor_state).squeeze(-1)
        return grip_logits, continuous_mean, value

    def sample_action(self, light_input, vibration_input, dt=None):
        grip_logits, continuous_mean, value = self.policy(
            light_input, vibration_input, dt
        )

        grip_distribution = Bernoulli(logits=grip_logits)
        grip_action = grip_distribution.sample()

        continuous_std = torch.exp(self.action_log_std).clamp(min=1e-4, max=2.0)
        continuous_distribution = Normal(continuous_mean, continuous_std)
        raw_continuous_action = continuous_distribution.sample()
        continuous_action = raw_continuous_action.clamp(0.0, 1.0)

        action = torch.cat([grip_action, continuous_action])
        log_prob = grip_distribution.log_prob(grip_action).sum()
        log_prob = (
            log_prob + continuous_distribution.log_prob(raw_continuous_action).sum()
        )
        entropy = (
            grip_distribution.entropy().sum() + continuous_distribution.entropy().sum()
        )

        return action, log_prob, entropy, value

    def deterministic_action(self, light_input, vibration_input, dt=None):
        grip_logits, continuous_mean, _ = self.policy(light_input, vibration_input, dt)
        grip_action = (torch.sigmoid(grip_logits) > 0.5).float()
        return torch.cat([grip_action, continuous_mean])

    def reset(self):
        with torch.no_grad():
            self.neuron_states.zero_()
