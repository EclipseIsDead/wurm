# wurm

Python simulation of C. elegans from an ML perspective: a 302-neuron recurrent brain controls a segmented worm body.

## Brain YAML

The brain and simulation constants live in `brain/brain.yaml`.

The YAML file stores:

- neuron counts for the 302-neuron model
- sensory split for 30 light sensors and 30 vibration sensors
- motor/action mapping for 2 grip actions plus 46 longitudinal/transverse segment controls
- recurrent dynamics initialization
- worm body constants
- foraging environment constants
- training hyperparameters

## Simulation

Run training with:

```bash
python -m simulation.main --episodes 10
```

Save a trained brain checkpoint with:

```bash
python -m simulation.main --episodes 200 --checkpoint checkpoints/wurm.pt
```

The loop is an actor-critic policy-gradient setup:

1. The environment emits light and vibration observations.
2. `CElegansBrain` updates recurrent neuron state.
3. The policy samples a mixed action:
   - Bernoulli head/tail grip actions
   - Normal continuous segment flex actions
4. The worm body applies those actions to grip, bend, and move.
5. Reward is based on food progress, food contact, action cost, grip cost, and wall contact.
6. The optimizer updates the recurrent brain, sensory scales, action variance, and value head.

## Pygame Test Harness

Open the visual simulator with:

```bash
python -m simulation.viewer
```

Load a trained checkpoint with:

```bash
python -m simulation.viewer --checkpoint checkpoints/wurm.pt --deterministic
```

The viewer runs the same `WormForagingEnv` used by training. Policy mode uses
the brain to act in the world; manual mode lets you directly test whether
head/tail gripping and segment flexion can move the body.

Runtime controls:

- `m`: switch policy/manual mode
- `r`: reset the episode
- `t`: switch sampled/deterministic policy actions
- arrow keys or WASD: drive the worm in manual mode
- mouse click: move the food
- Escape: quit
