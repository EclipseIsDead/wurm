# wurm
Python simulation of C. elegans, via 302 neurons and reinforcment learning.

## Worm Stimuli
- Light reception
- Vibration sensation

## Worm Actions
- Grip head/tail
- Flex muscle segment longitudinally (contract/extend)
- Flex muscle segment transversely (left/right)

## RL
- Proximal Policy Optimization (PPO)
- Deep Q Network (DQN)

Note the reward here should be related to food.

# Some things to keep in mind:
- The head/tail gripping problem is a binary classification issue, while the segment flexing is a continuous one
  - Seperate networks? How does this present biologically?
