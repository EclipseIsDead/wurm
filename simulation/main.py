import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import CElegansBrain, DEFAULT_CONFIG_PATH, load_config
from simulation.checkpoint import save_brain_checkpoint
from simulation.environment import WormForagingEnv


def discounted_returns(rewards, gamma, device):
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = float(reward) + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def train(config_path=DEFAULT_CONFIG_PATH, episodes=None, checkpoint=None):
    config = load_config(config_path)
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 7))
    torch.manual_seed(seed)
    np.random.seed(seed)

    brain = CElegansBrain(config=config)
    env = WormForagingEnv(config)
    optimizer = torch.optim.Adam(
        brain.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
    )

    episode_count = int(
        episodes if episodes is not None else training_config.get("episodes", 200)
    )
    max_steps = int(training_config.get("max_steps_per_episode", 200))
    gamma = float(training_config.get("gamma", 0.97))
    value_loss_weight = float(training_config.get("value_loss_weight", 0.5))
    entropy_weight = float(training_config.get("entropy_weight", 0.001))
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))
    log_every = int(training_config.get("log_every", 10))
    metrics = []

    for episode in range(1, episode_count + 1):
        observation = env.reset(seed + episode)
        brain.reset()
        log_probs = []
        values = []
        entropies = []
        rewards = []
        info = {}

        for _ in range(max_steps):
            action, log_prob, entropy, value = brain.sample_action(
                observation["light"],
                observation["vibration"],
            )
            observation, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            rewards.append(reward)
            if done:
                break

        returns = discounted_returns(rewards, gamma, values[0].device)
        value_tensor = torch.stack(values)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)

        advantages = returns - value_tensor.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        policy_loss = -(log_prob_tensor * advantages).mean()
        value_loss = F.mse_loss(value_tensor, returns)
        entropy_bonus = entropy_tensor.mean()
        loss = (
            policy_loss
            + value_loss_weight * value_loss
            - entropy_weight * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), grad_clip_norm)
        optimizer.step()

        total_reward = float(np.sum(rewards))
        episode_metrics = {
            "episode": episode,
            "reward": total_reward,
            "steps": len(rewards),
            "loss": float(loss.detach().cpu()),
            "distance_to_food": float(
                info.get("distance_to_food", env.distance_to_food())
            ),
            "reached_food": bool(info.get("reached_food", False)),
        }
        metrics.append(episode_metrics)

        if episode == 1 or episode % log_every == 0 or episode == episode_count:
            print(
                "episode={episode:04d} reward={reward:8.3f} steps={steps:03d} "
                "loss={loss:8.3f} distance={distance_to_food:7.2f} reached={reached_food}".format(
                    **episode_metrics
                )
            )

    if checkpoint is not None:
        save_brain_checkpoint(brain, checkpoint)

    return brain, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the YAML-configured worm brain."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the brain YAML file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override the episode count from the YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path where the trained brain checkpoint should be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, episodes=args.episodes, checkpoint=args.checkpoint)
