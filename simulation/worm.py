import numpy as np
import torch


class Worm:
    # 94 motor neurons -> 2 grip actions and 46 * 2 segment controls.
    def __init__(
        self,
        length=46,
        initial_position=(400, 300),
        segment_length=10,
        segment_width=5,
        movement_scale=8.0,
        turn_scale=0.35,
    ):
        self.length = int(length)
        self.default_segment_length = float(segment_length)
        self.default_segment_width = float(segment_width)
        self.movement_scale = float(movement_scale)
        self.turn_scale = float(turn_scale)
        self.grip_threshold = 0.5
        self.min_segment_length = self.default_segment_length * 0.5
        self.max_segment_length = self.default_segment_length * 1.5
        self.min_segment_width = max(1.0, self.default_segment_width * 0.5)
        self.max_segment_width = self.default_segment_width * 1.5
        self.segments = []
        self.reset(initial_position)

    def reset(self, initial_position=(400, 300), heading=0.0):
        self._head_position = np.array(initial_position, dtype=float)
        self.heading = float(heading)
        self._segment_lengths = np.full(self.length, self.default_segment_length)
        self._segment_widths = np.full(self.length, self.default_segment_width)
        self._segment_bends = np.zeros(self.length)
        self.head_fixed = False
        self.tail_fixed = False
        self.last_displacement = np.zeros(2)
        self._rebuild_segments()

    @property
    def head_position(self):
        return self._head_position.copy()

    @property
    def tail_position(self):
        return np.array(self.segments[-1]["position"], dtype=float)

    @property
    def speed(self):
        return float(np.linalg.norm(self.last_displacement))

    def set_head_position(self, position):
        self._head_position = np.array(position, dtype=float)
        self._rebuild_segments()

    def move(self, motor_input):
        if isinstance(motor_input, torch.Tensor):
            motor_input = motor_input.detach().cpu().numpy()
        else:
            motor_input = np.array(motor_input)
        motor_input = motor_input.astype(float).reshape(-1)

        expected_actions = 2 + self.length * 2
        if motor_input.size != expected_actions:
            raise ValueError(f"expected {expected_actions} motor actions, got {motor_input.size}")

        self.head_fixed = motor_input[0] > self.grip_threshold
        self.tail_fixed = motor_input[1] > self.grip_threshold

        flex = np.clip(motor_input[2:], 0.0, 1.0).reshape(self.length, 2)
        longitudinal = flex[:, 0] - 0.5
        transverse = flex[:, 1] - 0.5

        target_lengths = self.default_segment_length * (1.0 + 0.7 * longitudinal)
        target_widths = self.default_segment_width * (1.0 + 0.7 * transverse)
        self._segment_lengths = np.clip(
            0.85 * self._segment_lengths + 0.15 * target_lengths,
            self.min_segment_length,
            self.max_segment_length,
        )
        self._segment_widths = np.clip(
            0.85 * self._segment_widths + 0.15 * target_widths,
            self.min_segment_width,
            self.max_segment_width,
        )
        self._segment_bends = 0.8 * self._segment_bends + 0.2 * transverse

        self.heading = self._wrap_angle(
            self.heading + float(np.mean(self._segment_bends)) * self.turn_scale
        )
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])

        anchor_direction = float(self.tail_fixed) - float(self.head_fixed)
        if self.head_fixed and self.tail_fixed:
            stride = 0.0
        elif anchor_direction != 0.0:
            stride = anchor_direction * float(np.mean(np.abs(longitudinal)))
        else:
            stride = 0.2 * float(np.mean(longitudinal))

        self.last_displacement = direction * stride * self.movement_scale
        self._head_position = self._head_position + self.last_displacement
        self._rebuild_segments()

    def _rebuild_segments(self):
        self.segments = []
        position = self._head_position.copy()
        angle = self.heading

        for i in range(self.length):
            self.segments.append(
                {
                    "position": position.tolist(),
                    "length": float(self._segment_lengths[i]),
                    "width": float(self._segment_widths[i]),
                }
            )
            if i < self.length - 1:
                angle = self._wrap_angle(angle + float(self._segment_bends[i]) * 0.08)
                direction = np.array([np.cos(angle), np.sin(angle)])
                position = position - direction * self._segment_lengths[i]

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def draw(self, screen):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("pygame is required to draw the worm") from exc

        for segment in self.segments:
            pos = segment["position"]
            length = segment["length"]
            width = segment["width"]
            pygame.draw.ellipse(
                screen,
                (255, 0, 0),
                (pos[0] - length / 2, pos[1] - width / 2, length, width),
            )
