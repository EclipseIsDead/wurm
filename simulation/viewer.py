import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import CElegansBrain, DEFAULT_CONFIG_PATH, load_config
from simulation.checkpoint import load_brain_checkpoint
from simulation.environment import WormForagingEnv


BACKGROUND = (16, 18, 22)
WALL = (64, 68, 78)
FOOD = (86, 190, 114)
FOOD_RING = (170, 230, 150)
WORM = (222, 82, 76)
WORM_HEAD = (255, 196, 95)
WORM_TAIL = (146, 170, 255)
TRACE = (52, 94, 128)
TEXT = (226, 232, 240)
MUTED = (132, 145, 160)


class PygameWormViewer:
    def __init__(
        self,
        config,
        checkpoint=None,
        manual=False,
        deterministic=False,
        steps_per_frame=1,
        fps=60,
    ):
        self.config = config
        self.env = WormForagingEnv(config)
        self.brain = CElegansBrain(config=config)
        if checkpoint is not None:
            load_brain_checkpoint(self.brain, checkpoint)

        self.manual = bool(manual)
        self.deterministic = bool(deterministic)
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.fps = max(1, int(fps))
        self.episode = 1
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {}
        self.trace = []
        self.observation = self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_info = {}
        self.trace = []
        self.brain.reset()
        return self.env.reset()

    def run(self, max_frames=None):
        import pygame

        pygame.init()
        world_size = self.env.world_size.astype(int)
        screen = pygame.display.set_mode((int(world_size[0]), int(world_size[1])))
        pygame.display.set_caption("Wurm")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("menlo", 15)
        small_font = pygame.font.SysFont("menlo", 12)

        running = True
        frame = 0
        while running:
            frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key, pygame)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.env.set_food_position(event.pos)

            keys = pygame.key.get_pressed()
            for _ in range(self.steps_per_frame):
                self._step(keys, pygame)

            self._draw(screen, font, small_font, pygame)
            pygame.display.flip()
            clock.tick(self.fps)

            if max_frames is not None and frame >= max_frames:
                running = False

        pygame.quit()

    def _handle_key(self, key, pygame):
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self.observation = self.reset()
        elif key == pygame.K_m:
            self.manual = not self.manual
        elif key == pygame.K_t:
            self.deterministic = not self.deterministic
        return True

    def _step(self, keys, pygame):
        if self.manual:
            action = self._manual_action(keys, pygame)
        else:
            action = self._policy_action()

        self.observation, reward, done, info = self.env.step(action)
        self.last_reward = reward
        self.total_reward += reward
        self.last_info = info
        self.trace.append(self.env.worm.head_position)
        if len(self.trace) > 220:
            self.trace.pop(0)

        if done:
            self.episode += 1
            self.observation = self.reset()

    def _policy_action(self):
        with torch.no_grad():
            if self.deterministic:
                return self.brain.deterministic_action(
                    self.observation["light"],
                    self.observation["vibration"],
                )
            action, _, _, _ = self.brain.sample_action(
                self.observation["light"],
                self.observation["vibration"],
            )
            return action

    def _manual_action(self, keys, pygame):
        action = np.full(2 + self.env.worm.length * 2, 0.5, dtype=np.float32)
        action[0:2] = 0.0

        moving_forward = keys[pygame.K_UP] or keys[pygame.K_w]
        moving_backward = keys[pygame.K_DOWN] or keys[pygame.K_s]
        turning_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        turning_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        if moving_forward:
            action[1] = 1.0
            action[2::2] = 1.0
        elif moving_backward:
            action[0] = 1.0
            action[2::2] = 1.0

        if turning_left and not turning_right:
            action[3::2] = 0.0
        elif turning_right and not turning_left:
            action[3::2] = 1.0

        return action

    def _draw(self, screen, font, small_font, pygame):
        screen.fill(BACKGROUND)
        self._draw_world(screen, pygame)
        self._draw_trace(screen, pygame)
        self._draw_food(screen, pygame)
        self._draw_worm(screen, pygame)
        self._draw_hud(screen, font, small_font)

    def _draw_world(self, screen, pygame):
        rect = pygame.Rect(
            0,
            0,
            int(self.env.world_size[0]),
            int(self.env.world_size[1]),
        )
        pygame.draw.rect(screen, WALL, rect, width=2)

    def _draw_trace(self, screen, pygame):
        if len(self.trace) < 2:
            return
        points = [tuple(point.astype(int)) for point in self.trace]
        pygame.draw.lines(screen, TRACE, False, points, width=2)

    def _draw_food(self, screen, pygame):
        center = tuple(self.env.food_position.astype(int))
        pygame.draw.circle(
            screen,
            FOOD_RING,
            center,
            int(self.env.food_radius + 7),
            width=1,
        )
        pygame.draw.circle(screen, FOOD, center, int(self.env.food_radius))

    def _draw_worm(self, screen, pygame):
        segments = self.env.worm.segments
        if len(segments) > 1:
            spine = [
                (int(segment["position"][0]), int(segment["position"][1]))
                for segment in segments
            ]
            pygame.draw.lines(screen, (115, 48, 46), False, spine, width=3)

        for index, segment in enumerate(reversed(segments)):
            position = segment["position"]
            length = segment["length"]
            width = segment["width"]
            color = WORM
            if index == len(segments) - 1:
                color = WORM_HEAD
            elif index == 0:
                color = WORM_TAIL
            rect = pygame.Rect(
                int(position[0] - length / 2),
                int(position[1] - width / 2),
                max(2, int(length)),
                max(2, int(width)),
            )
            pygame.draw.ellipse(screen, color, rect)

        head = self.env.worm.head_position
        heading = self.env.worm.heading
        nose = head + np.array([np.cos(heading), np.sin(heading)]) * 18.0
        pygame.draw.line(
            screen,
            WORM_HEAD,
            tuple(head.astype(int)),
            tuple(nose.astype(int)),
            width=3,
        )

    def _draw_hud(self, screen, font, small_font):
        mode = "manual" if self.manual else "policy"
        sampling = "deterministic" if self.deterministic else "sampled"
        reached = self.last_info.get("reached_food", False)
        distance = self.last_info.get(
            "distance_to_food",
            self.env.distance_to_food(),
        )
        lines = [
            f"{mode} / {sampling}",
            f"episode {self.episode}  step {self.env.steps}/{self.env.max_steps}",
            f"reward {self.total_reward:7.2f}  last {self.last_reward:6.2f}",
            f"distance {distance:7.2f}  reached {reached}",
        ]

        x = 14
        y = 12
        for index, line in enumerate(lines):
            surface = font.render(line, True, TEXT if index == 0 else MUTED)
            screen.blit(surface, (x, y))
            y += 20

        if self.manual:
            if self.env.worm.head_fixed:
                grip = "head grip"
            elif self.env.worm.tail_fixed:
                grip = "tail grip"
            else:
                grip = "free"
            surface = small_font.render(grip, True, MUTED)
            screen.blit(surface, (x, y + 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Open the Pygame worm simulator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the brain YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained brain checkpoint to load.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Start with keyboard-driven body controls.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use action means instead of sampled actions in policy mode.",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=1,
        help="Simulation steps to run each rendered frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Render frame rate.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit after this many frames; useful for smoke tests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    viewer = PygameWormViewer(
        config,
        checkpoint=args.checkpoint,
        manual=args.manual,
        deterministic=args.deterministic,
        steps_per_frame=args.steps_per_frame,
        fps=args.fps,
    )
    viewer.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
