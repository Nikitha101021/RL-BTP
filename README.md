# Hybrid Autonomous Driving Framework using Reinforcement Learning and Indian Road Perception

## Overview
This repository is a professional B.Tech research framework that leverages Reinforcement Learning (PPO) combined with Computer Vision (Perception) for autonomous driving. It transitions from a single/multi-agent fixed-track racing setup into a scalable autonomous driving simulation and evaluation framework.

## Key Features
- **Modular Architecture:** Clear separation between Perception, RL, Evaluation, Datasets, and Configurations.
- **Reinforcement Learning:** PPO with CNN Policy capable of handling continuous control.
- **Custom Environments:** Single-agent and multi-agent dynamic traffic scenarios.
- **Extensible Perception Hooks:** Prepared modules for YOLO Object Detection, Lane Detection, and Road Segmentation (Indian Driving Dataset compatible).
- **Configurable Rewards:** Default, custom, and risk-aware reward models.

## Structure
```
project/
├── perception/      # Object detection, Lane detection
├── rl/              # PPO algorithms, environments, custom reward structures
├── evaluation/      # Metric calculation and test scripts
├── datasets/        # IDD, KITTI, Mapillary
├── models/          # Trained PPO models
├── checkpoints/     # Intermediate model states
├── outputs/         # Saved logs, graphs, videos, and screenshots
├── utils/           # Helper scripts
└── configs/         # Centralized hyperparameters
```

## Setup & Training
All hyperparameters are managed via `configs/`.

To train the autonomous driving model:
```bash
python -m rl.ppo.train
```

To evaluate a trained model:
```bash
python -m evaluation.autonomous_drive_test
```
