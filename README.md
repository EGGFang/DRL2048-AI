# DRL2048-AI
A Deep Reinforcement Learning Agent for Playing 2048

## Introduction
This project features an AI that learns to play the 2048 game using Deep Reinforcement Learning (DRL). A custom 2048 environment was built using pygame and designed to be gym-compatible, supporting AI training and testing. The environment allows for standard action inputs (left, right, up, down) and provides essential game state information, including tile values, rewards, and game termination conditions.

The AI was trained using three different DRL methods: A2C, D3QN, and DDQN, with DDQN achieving the best results. A graphical interface was also implemented to visualize the AI's gameplay.

## Environment
![image](https://github.com/user-attachments/assets/d2905c5a-3b6c-4c40-9c1c-1ab2ed656930)

* Implemented using pygame, fully compatible with OpenAI Gym API format.
* Supports four discrete actions (left, right, up, down).
* Provides game state observations including:
* - Current board tile values
* - Reward after each move
* - Game-over status
* Includes a graphical interface for real-time AI gameplay visualization.

## Methods
Three Deep Reinforcement Learning approaches were explored:
![image](https://github.com/user-attachments/assets/3e1073f8-cb17-441a-b8e8-91752fa1060b)

* A2C (Advantage Actor-Critic)
* D3QN (Double Dueling Deep Q-Network)
* DDQN (Double Deep Q-Network) – Best performing method
The AI model was trained for 100,000 episodes. The first 2048 tile was achieved after ~60,000 episodes, and became more frequent after 90,000 episodes.

## Results
The final trained model demonstrated consistent performance:
![image](https://github.com/user-attachments/assets/10e6627f-6984-4e0d-b09e-ae6bd2c48173)

* The AI reached at least 512 in 75.52% of games.
* The AI reached at least 1024 in 34.82% of games.
* The AI reached 2048 in 1.21% of games.
The results confirm that the AI can play 2048 effectively, achieving high scores with a structured strategy. More details can be seen in the visual results below:
