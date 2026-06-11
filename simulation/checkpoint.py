from pathlib import Path

import torch


def save_brain_checkpoint(brain, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": brain.config,
            "state_dict": brain.state_dict(),
        },
        path,
    )


def load_brain_checkpoint(brain, path):
    checkpoint = torch.load(Path(path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    brain.load_state_dict(state_dict)
    return checkpoint
