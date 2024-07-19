import pygame
import torch
import numpy as np

class Worm:
    # 94 motor neurons -> 1 for head 1 for tail, 92/2 segments
    def __init__(self, length=46, initial_position=(400, 300), segment_length=10, segment_width=5):
        self.length = length
        self.segments = []

        # segment init (independent)
        for i in range(length):
            x = initial_position[0] + i * segment_length
            y = initial_position[1]
            self.segments.append({
                'position': [x, y],
                'length': segment_length,
                'width': segment_width
            })

        # worm has hairs on head and tail to affix position
        self.head_fixed = False
        self.tail_fixed = False

    def move(self, motor_input):
        # lol
        if isinstance(motor_input, torch.Tensor):
            motor_input = motor_input.detach().cpu().numpy()
        else:
            motor_input = np.array(motor_input)

        # binary classification for head and tail
        self.head_fixed = motor_input[0] > 0.5
        self.tail_fixed = motor_input[-1] > 0.5

        # segment movement head to tail
        for i in range(self.length):
            if (i == 0 and self.head_fixed) or (i == self.length - 1 and self.tail_fixed):
                continue

            length_change = motor_input[i*2] - 0.5  # -0.5 to 0.5
            width_change = motor_input[i*2 + 1] - 0.5  # ^

            # update
            self.segments[i]['length'] += length_change
            self.segments[i]['width'] += width_change

            # double check scaling
            self.segments[i]['length'] = max(self.segments[i]['length'], 1)
            self.segments[i]['width'] = max(self.segments[i]['width'], 1)

        for i in range(1, self.length):
            prev_segment = self.segments[i-1]
            curr_segment = self.segments[i]
            angle = np.arctan2(curr_segment['position'][1] - prev_segment['position'][1],
                               curr_segment['position'][0] - prev_segment['position'][0])
            new_x = prev_segment['position'][0] + np.cos(angle) * prev_segment['length']
            new_y = prev_segment['position'][1] + np.sin(angle) * prev_segment['length']

            curr_segment['position'] = [new_x, new_y]

    def draw(self, screen):
        for segment in self.segments:
            pos = segment['position']
            length = segment['length']
            width = segment['width']
            # this is probably wrong drawing?
            pygame.draw.ellipse(screen, (255, 0, 0),
                                (pos[0] - length/2, pos[1] - width/2, length, width))
