import numpy as np

from simulation.worm import Worm


class WormForagingEnv:
    def __init__(self, config):
        self.config = config
        brain_config = config["brain"]
        worm_config = config.get("worm", {})
        env_config = config.get("environment", {})

        self.world_size = np.array(
            env_config.get("world_size", [800.0, 800.0]),
            dtype=float,
        )
        self.food_position_config = np.array(
            env_config.get("food_position", [620.0, 400.0]),
            dtype=float,
        )
        self.randomize_food = bool(env_config.get("randomize_food", False))
        self.food_radius = float(env_config.get("food_radius", 18.0))
        self.food_reward = float(env_config.get("food_reward", 10.0))
        self.progress_reward_scale = float(
            env_config.get("progress_reward_scale", 4.0)
        )
        self.step_penalty = float(env_config.get("step_penalty", 0.01))
        self.control_penalty = float(env_config.get("control_penalty", 0.01))
        self.grip_penalty = float(env_config.get("grip_penalty", 0.005))
        self.wall_penalty = float(env_config.get("wall_penalty", 0.2))
        self.light_distance_scale = float(env_config.get("light_distance_scale", 260.0))
        self.vibration_distance_scale = float(
            env_config.get("vibration_distance_scale", 120.0)
        )
        self.num_light_sensors = int(brain_config["num_light_sensors"])
        self.num_vibration_sensors = int(brain_config["num_vibration_sensors"])
        self.max_steps = int(
            config.get("training", {}).get("max_steps_per_episode", 200)
        )
        self.rng = np.random.default_rng(
            int(config.get("training", {}).get("seed", 7))
        )

        self.initial_position = np.array(
            worm_config.get("initial_position", [400.0, 400.0]),
            dtype=float,
        )
        self.worm = Worm(
            length=int(worm_config.get("segments", 46)),
            initial_position=self.initial_position,
            segment_length=float(worm_config.get("segment_length", 10.0)),
            segment_width=float(worm_config.get("segment_width", 5.0)),
            movement_scale=float(worm_config.get("movement_scale", 8.0)),
            turn_scale=float(worm_config.get("turn_scale", 0.35)),
        )

        expected_actions = 2 + self.worm.length * 2
        if expected_actions != int(brain_config["num_motor_neurons"]):
            raise ValueError(
                "worm action count must match motor neurons "
                f"({expected_actions} != {brain_config['num_motor_neurons']})"
            )

        self.steps = 0
        self.food_position = self.food_position_config.copy()
        self.previous_distance = self.distance_to_food()
        self.last_wall_hit = False

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.last_wall_hit = False
        self.worm.reset(self.initial_position, heading=0.0)
        self.food_position = self._new_food_position()
        self.previous_distance = self.distance_to_food()
        return self.observation()

    def step(self, action):
        self.steps += 1
        previous_distance = self.distance_to_food()

        action = self._action_array(action)
        self.worm.move(action)
        wall_hit = self._keep_head_in_world()
        current_distance = self.distance_to_food()
        progress = previous_distance - current_distance

        continuous = action[2:] - 0.5
        reward = self.progress_reward_scale * progress
        reward -= self.step_penalty
        reward -= self.control_penalty * float(np.mean(np.square(continuous)))
        reward -= self.grip_penalty * float(action[0] + action[1])

        reached_food = current_distance <= self.food_radius
        if reached_food:
            reward += self.food_reward
        if wall_hit:
            reward -= self.wall_penalty

        done = reached_food or self.steps >= self.max_steps
        self.previous_distance = current_distance

        info = {
            "distance_to_food": current_distance,
            "progress": progress,
            "reached_food": reached_food,
            "wall_hit": wall_hit,
            "head_position": self.worm.head_position,
            "food_position": self.food_position.copy(),
        }
        return self.observation(), float(reward), done, info

    def observation(self):
        distance, relative_angle = self._food_polar()
        light = self._directional_sensor(
            self.num_light_sensors,
            relative_angle,
            np.exp(-distance / self.light_distance_scale),
        )

        center = self.world_size / 2.0
        to_center = center - self.worm.head_position
        center_angle = np.arctan2(to_center[1], to_center[0]) - self.worm.heading
        wall_distance = self._wall_distance()
        wall_gain = np.exp(-wall_distance / self.vibration_distance_scale)
        motion_gain = min(1.0, self.worm.speed / max(self.worm.movement_scale, 1e-6))
        vibration = self._directional_sensor(
            self.num_vibration_sensors,
            center_angle,
            0.7 * wall_gain + 0.3 * motion_gain,
        )

        return {
            "light": light.astype(np.float32),
            "vibration": vibration.astype(np.float32),
        }

    def distance_to_food(self):
        return float(np.linalg.norm(self.food_position - self.worm.head_position))

    def set_food_position(self, position):
        self.food_position = np.clip(
            np.asarray(position, dtype=float),
            [0.0, 0.0],
            self.world_size,
        )
        self.previous_distance = self.distance_to_food()

    def _new_food_position(self):
        if not self.randomize_food:
            return self.food_position_config.copy()

        low = self.world_size * 0.15
        high = self.world_size * 0.85
        for _ in range(100):
            candidate = self.rng.uniform(low, high)
            if (
                np.linalg.norm(candidate - self.initial_position)
                > self.world_size.min() * 0.2
            ):
                return candidate
        return self.food_position_config.copy()

    def _food_polar(self):
        vector = self.food_position - self.worm.head_position
        distance = float(np.linalg.norm(vector))
        angle = np.arctan2(vector[1], vector[0]) - self.worm.heading
        return distance, self._wrap_angle(angle)

    def _directional_sensor(self, count, relative_angle, gain):
        sensor_angles = np.linspace(-np.pi, np.pi, count, endpoint=False)
        response = np.maximum(0.0, np.cos(sensor_angles - relative_angle))
        return np.clip(response * gain, 0.0, 1.0)

    def _wall_distance(self):
        head = self.worm.head_position
        return float(
            min(
                head[0],
                head[1],
                self.world_size[0] - head[0],
                self.world_size[1] - head[1],
            )
        )

    def _keep_head_in_world(self):
        head = self.worm.head_position
        clipped = np.clip(head, [0.0, 0.0], self.world_size)
        wall_hit = not np.allclose(head, clipped)
        if wall_hit:
            self.worm.set_head_position(clipped)
        self.last_wall_hit = wall_hit
        return wall_hit

    def _action_array(self, action):
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=float).reshape(-1)

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
