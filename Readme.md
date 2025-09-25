# Gin Rummy RL Training

This repository implements adversarial RL training for a 2-player Gin Rummy game. We use PettingZoo's environment for the game logic, ma-gym for multi-agent wrappers if needed, and Stable-Baselines3 to train an RL agent (e.g., PPO) against a fixed LLM enhancer opponent. The focus is on reinforcing the RL model through iterative play to achieve high win rates.
Gin Rummy involves a 52-card deck where players draw and discard to form melds (3+ cards of same rank or sequence in suit), aiming to minimize deadwood points. Knock with ≤10 deadwood or go gin (0 deadwood) to score against the opponent.

---

## Requirements
- Python 3.8+
- `pettingzoo[classic]`
- `ma-gym`
- `stable-baselines3`
- `gymnasium`

---

## Installation
```bash
pip install pettingzoo[classic] ma-gym stable-baselines3 gymnasium
```

---

## Usage
Set up the environment:
```python
from pettingzoo.classic import gin_rummy_v4
env = gin_rummy_v4.env()
```

Train RL agent (PPO):
```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("gin_rummy_rl")
```

See `train.py` for details.

---

## Contributing
Contributions welcome! Fork, improve (e.g., LLM integration), and submit PRs.

---

## Description


Vision-language models (VLMs) and their extensions, such as vision-language-action (VLA) models, represent a transformative intersection of computer vision (CV) and natural language processing (NLP), enabling agents to interpret visual data through linguistic reasoning. In the context of multiplayer games like Gin Rummy, these models facilitate tasks requiring real-time decision-making, such as strategy optimization, card melding, and adversarial play. This survey synthesizes recent advancements up to 2025, drawing from key literature on model architectures, optimization strategies, datasets, and applications. It expands on the project README by providing in-depth technical details, feasibility considerations for student projects, and broader implications, while emphasizing balanced views on challenges like computational overhead and ethical concerns.
Evolution and Key Concepts of VLMs and VLA Models in Games
VLMs integrate visual encoders (e.g., Vision Transformers or ViTs) with language models (e.g., transformers like LLaMA) to process multimodal inputs. Early models like CLIP (2021) focused on image-text alignment, but by 2025, advancements include unified tokenization for vision, language, and actions, enabling end-to-end reasoning. VLA models extend this by incorporating action generation, often via policy modules for robotic or game control.
Recent phases (2022-2025) show progression: foundational integration (e.g., RT-1 for visuomotor tasks), specialization (e.g., domain-specific biases in 2024), and generalization with safety (e.g., SafeVLA in 2025). Core techniques include cross-attention for fusion, token unification, and affordance detection. For games like Gin Rummy, methods like SPAG use retrieval-augmented generation (RAG) pipelines to incorporate domain knowledge, enhancing strategic understanding in competitive settings.
Controversies arise around bias in training data—e.g., web-crawled datasets may underrepresent diverse game scenarios, leading to uneven performance across player strategies. Research suggests countering this through balanced datasets and fine-tuning, though evidence leans toward proprietary models outperforming open-source in robustness, per 2025 benchmarks.
