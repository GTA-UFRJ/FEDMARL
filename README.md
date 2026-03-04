# ClientSelection — Federated Client Selection using Multi-Agent Reinforcement Learning

This repository contains the implementation of a client selection mechanism for *Federated Learning (FL)* based on *Multi-Agent Reinforcement Learning (MARL)*, using *Value Decomposition Networks (VDN)*, with a focus on robustness against label flipping attacks.

---

## Project Description

In federated learning scenarios, the selection of clients participating in each aggregation round directly impacts the quality and robustness of the global model. Malicious clients can degrade model performance by sending poisoned updates.

This project proposes the use of **multi-agent reinforcement learning** agents to adaptively select clients, avoiding attackers and prioritizing honest clients based on contribution metrics.

### Main Components

- **VDN (Value Decomposition Networks)** with Double DQN and Prioritized Experience Replay (PER) for client selection
- **Agent state metrics**: gradient projection (proj), generalization loss (gener), staleness (estag) and selection streak (serie)
- **Attack**: Deterministic Targeted Label Flipping with configurable attacker fraction
- **Aggregation mechanism**: FedAvg 
- **Data distribution**: Non-IID Dirichlet split with configurable alpha

### Experiment Architecture

Each round is divided into two phases:

1. **Metrics phase** — all 50 clients train for `local_steps`. The metrics (`proj`, `gener`) are computed and used as state variables.
2. **Training phase** — only the K clients selected by the agent train for `local_epochs`. The deltas are aggregated via FedAvg.

---

## Agent State Metrics

The local observation vector of each client $i$ at round $t$ is composed of the following metrics:

**Gradient projection**:

$$\text{proj}_{i,t} = \Delta w_i^\top \cdot \hat{m}_t$$

$$m_t = \beta m_{t-1} + (1-\beta)\nabla_{w_t}\mathcal{L}(w_t; \mathcal{D}^{val}), \qquad \hat{m}_t = \frac{-m_t}{\|m_t\| + \epsilon}$$

**Generalization loss**:

$$\text{gener}_{i,t} = \frac{1}{|\mathcal{D}|}\sum_{j=1}^{|\mathcal{D}|} \mathcal{L}\left(\hat{y}_{i,t}^{(j)}, y_{i,t}^{(j)}\right)$$

**Staleness**:

$$\text{estag}^{\ast}_{i,t} = \frac{\text{estag}_{i,t}}{\max_{j \neq i}\, \text{estag}_{j,t} + \epsilon}$$

**Selection streak**:

$$\text{serie}^{\ast}_{i,t} = \min\left(\frac{\text{serie}_{i,t}}{\text{serie}^{(\max)}}, 1\right)$$

---

## Attack: Label Flipping

The implemented attack is a **targeted label flipping**, where each class is mapped
to a visually similar class following a fixed mapping:

| Original   | Flipped    |
|------------|---------   |
| airplane   | ship       |
| ship       | airplane   |
| automobile | truck      |
| truck      | automobile |
| cat        | dog        |
| dog        | cat        |
| deer       | horse      |
| horse      | deer       |
| bird       | frog       |
| frog       | bird       |

Unlike random flipping, this approach is more realistic and harder to detect,
as the model learns confusions between visually similar classes. The `attack_rate`
parameter controls the fraction of samples flipped per attacking client.

---

## Installation

### Clone the repository
```bash
git clone https://github.com/GTA-UFRJ/FEDMARL.git
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Default execution
```bash
python main.py
```

### Main configuration (`main.py`)

The main hyperparameters are passed directly to `run_experiment()`:
```python
run_experiment(
    rounds=350,
    n_clients=50,
    k_select=15,
    dir_alpha=0.3,
    run_random=True,            # runs random selection track
    run_vdn=True,               # runs VDN track
    initial_flip_fraction=0.4,
    flip_rate_initial=1.0,
    local_lr=0.005,
    local_steps=10,
    local_epochs=5,
    marl_lr=1e-4,
)
```

Results are automatically saved to a `.json` file in the configured output directory.

---

## File Structure
```
ClientSelection/
+-- main.py           # entry point, hyperparameter configuration
+-- experiment.py     # main experiment loop (RANDOM and VDN tracks)
+-- server.py         # local training, aggregation, server metrics
+-- agent.py          # VDNSelector, AgentMLP, PrioritizedReplayJoint
+-- metrics.py        # eval_acc, eval_loss, probing_loss, windowed_reward
+-- data.py           # Dirichlet split and label flipping dataset
+-- model.py          # ResNet18 adapted for CIFAR-10
+-- config.py         # DEVICE, SEED, seed_worker
+-- flower/           # experimental implementation with Flower 1.26 (in development)
    +-- pyproject.toml
    +-- vdn_fl/
        +-- client_app.py
        +-- server_app.py
        +-- data.py
        +-- ...
```

---

## Model Evolution

The project went through three main development stages, each revealing limitations and motivating the following improvements. The examples below adopt the same base configuration: N = 50 clients, K = 15 selected per round, 40% attacking clients with full label inversion (100% label flipping).

### Stage 1 — SmallCNN

The initial version used a simple CNN (SmallCNN):

| Layer | Configuration |
|---|---|
| Input | Conv(3,3,32) + Pool(2×2) |
| Layer 2 | Conv(3,32,64) + Pool(2×2) |
| Layer 3 | Conv(3,64,128) + Pool(2×2) |
| Output | FC(2048, 256, 10) |
| Optimizer | SGD (momentum=0.9, lr=0.01) |

With this architecture the VDN agent already demonstrated superiority over random selection (FedAvg), reaching ~67% accuracy against ~55% for FedAvg with 40% attacking clients over 500 rounds. The small network, by generating lower magnitude deltas, exhibited natural stability against attacks.

![SmallCNN — FedAvg vs MARL with 40% attackers](assets/smallcnn_results.png)

---

### Stage 2 — ResNet18 without stabilization mechanisms

Replacing with ResNet18 adapted for CIFAR-10 (3×3 conv1, no maxpool, standard BatchNorm) aimed to increase model capacity and bring results closer to state of the art. However, without stabilization mechanisms, the higher magnitude deltas from ResNet18 drastically amplified the impact of attackers, causing sharp and recurrent accuracy drops that made training unstable.

![ResNet18 without norm filtering or clipping — severe oscillations](assets/resnet_no_defense.png)

---

### Stage 3 — ResNet18 with FedMedian + Norm Filtering + Clipping

Adding three mechanisms to the aggregation resolved the instability:

|   Mechanism      |       Configuration      |                         Effect                         |
|------------------|--------------------------|--------------------------------------------------------|
|Norm filtering    | `2.0 × median_norm`      | Discards deltas with anomalous norm before aggregation |
|Gradient clipping | `0.25 × median_norm`     | Limits the total update magnitude per round            |
|FedMedian         |           —              | Aggregates by coordinate-wise median                   |

With Those mechanisms, the VDN agent maintains stable accuracy around **85%** over 350 rounds while consistently selecting honest clients, whereas FedAvg with random selection oscillates continuously due to the presence of attackers.

![ResNet18 with FedMedian + norm filtering + clipping](assets/resnet_with_defense.png)

---

## Client Ranking

Every 20 rounds, the server prints the client ranking ordered by advantage (`adv = Q1 - Q0`). Clients with positive `adv` are prioritized for selection. The result below illustrates the separation learned by MARL, showing that the policy consistently ranks honest clients above attackers:

| Position  |   Client  | Type       |    adv    |
|---------- |  -------- |------------|-----------|
| 1st       |    41     |  HONEST    | +0.083774 |
| 2nd       |    06     |  HONEST    | +0.074431 |
| 3rd       |    23     |  HONEST    | +0.070137 |
| 4th       |    24     |  HONEST    | +0.068502 |
| ...       |   ...     |    ...     |     ...   |
| 47th      |    12     |  ATTACKER  | -0.144135 |
| 48th      |    31     |  ATTACKER  | -0.142081 |
| 49th      |    12     |  ATTACKER  | -0.144135 |
| 50th      |    30     |  ATTACKER  | -0.187876 |

---