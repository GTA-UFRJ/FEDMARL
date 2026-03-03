ClientSelection — Seleção de Clientes Federados usando Aprendizado por Reforço Multiagente

Este repositório contém a implementação de um mecanismo de seleção de clientes para **Aprendizado Federado (FL)** baseado em **Value Decomposition Networks (VDN)**, com foco em robustez contra ataques do tipo *label flipping*.



---

## Descrição do Projeto

Em cenários de aprendizado federado, a seleção dos clientes que participam de cada rodada de agregação impacta diretamente a qualidade e a robustez do modelo global. Clientes maliciosos (*atacantes bizantinos*) podem degradar o desempenho do modelo ao enviar atualizações envenenadas.

Este projeto propõe o uso de um agente de **aprendizado por reforço multi-agente** baseado em VDN para aprender a selecionar clientes de forma adaptativa, evitando atacantes e priorizando clientes honestos com base em métricas de qualidade de gradiente.

### Componentes principais

- **VDN (Value Decomposition Networks)** com Double DQN e Prioritized Experience Replay (PER) para seleção de clientes
- **Métricas de estado do agente**: projeção momentum (`proj_mom`), probing loss (`probe_now`), staleness e streak de seleção
- **Ataque**: *Targeted Label Flipping* determinístico com fração configurável de atacantes
- **Agregação robusta**: FedAvg
- **Distribuição de dados**: Dirichlet não-IID com alpha configurável

### Arquitetura do experimento

Cada rodada é dividida em duas fases:

1. **Fase de métricas** — todos os 50 clientes treinam por `local_steps` passos curtos. As métricas (`proj_mom`, `probe_now`, `fo`) são calculadas e enviadas ao agente.
2. **Fase de treino** — apenas os K clientes selecionados pelo agente treinam por `local_epochs` épocas completas. Os deltas são agregados via FedAvg
---

## Instalação

### Requisitos

- Python 3.11+
- PyTorch 2.0+
- CUDA (recomendado)

### Dependências

```bash
pip install torch torchvision numpy
```

### Clone o repositório

```bash
git clone https://github.com/braiton1277/ClientSelection.git
cd ClientSelection
```

---

## Como Rodar

### Execução padrão

```bash
python main.py
```

### Configuração principal (`main.py`)

Os principais hiperparâmetros são passados diretamente para `run_experiment()`:

```python
run_experiment(
    rounds=350,
    n_clients=50,
    k_select=15,
    dir_alpha=0.3,
    run_random=True,       # roda track de seleção aleatória
    run_vdn=True,          # roda track VDN
    initial_flip_fraction=0.4,
    flip_rate_initial=1.0,    
    flip_add_fraction=0.20,
    local_lr=0.005,
    local_steps=10,
    local_epochs=5,
    marl_lr=1e-4,
)
```

Os resultados são salvos automaticamente em um arquivo `.json` no diretório de saída configurado.

---

## Estrutura dos Arquivos

```
ClientSelection/
+-- main.py           # ponto de entrada, configuração dos hiperparâmetros
+-- experiment.py     # loop principal do experimento (tracks RANDOM e VDN)
+-- server.py         # treino local, agregação FedMedian, métricas de servidor
+-- agent.py          # VDNSelector, AgentMLP, PrioritizedReplayJoint
+-- metrics.py        # eval_acc, eval_loss, probing_loss, windowed_reward
+-- data.py           # split Dirichlet, SwitchableTargetedLabelFlipSubset
+-- model.py          # ResNet18 adaptada para CIFAR-10
+-- config.py         # DEVICE, SEED, seed_worker
+-- flower/           # implementação experimental com Flower 1.26 (em desenvolvimento)
    +-- pyproject.toml
    +-- vdn_fl/
        +-- client_app.py
        +-- server_app.py
        +-- data.py
        +-- ...
```

---

## Evolução do Modelo

O projeto passou por três etapas principais de desenvolvimento, cada uma evidenciando limitações e motivando as melhorias seguintes.

### Etapa 1 — SmallCNN

A versão inicial utilizava uma CNN simples (SmallCNN) com 4 camadas convolucionais:

| Camada | Configuração |
|---|---|
| Entrada | Conv(3,3,32) + Pool(2×2) |
| Camada 2 | Conv(3,32,64) + Pool(2×2) |
| Camada 3 | Conv(3,64,128) + Pool(2×2) |
| Saída | FC(2048, 256, 10) |
| Otimizador | SGD (momentum=0.9, lr=0.01) |

Com essa arquitetura o agente VDN já demonstrou superioridade sobre a seleção aleatória (FedAvg), atingindo ~67% de acurácia contra ~50% do FedAvg com 40% dos clientes atacantes ao longo de 500 rodadas. A rede pequena, por gerar deltas de menor magnitude, apresentava estabilidade natural contra ataques.

![SmallCNN — FedAvg vs MARL com 40% atacantes](assets/smallcnn_results.png)

---

### Etapa 2 — ResNet18 sem defesa

A substituição pela ResNet18 adaptada para CIFAR-10 (conv1 3×3, sem maxpool, BatchNorm padrão) visava aumentar a capacidade do modelo e aproximar os resultados do estado da arte (~85–88% sem ataque). Porém, sem mecanismos de defesa, os deltas de maior magnitude da ResNet18 amplificavam drasticamente o impacto dos atacantes, causando quedas bruscas e recorrentes de acurácia que tornavam o treinamento instável.

![ResNet18 sem norm filtering nem clipping — oscilações severas](assets/resnet_no_defense.png)

---

### Etapa 3 — ResNet18 com FedMedian + Norm Filtering + Clipping

A adição de três mecanismos de defesa na agregação resolveu a instabilidade:

| Mecanismo | Configuração | Efeito |
|---|---|---|
| Norm filtering | `2.0 × median_norm` | Remove deltas com norma anômala antes da agregação |
| Gradient clipping | `0.25 × median_norm` | Limita a magnitude total da atualização por rodada |
| FedMedian | — | Agrega pela mediana elemento a elemento, resistente a outliers |

Com essas defesas, o agente VDN mantém acurácia estável em torno de **80–85%** ao longo de 350 rodadas, enquanto o FedAvg com seleção aleatória oscila continuamente devido à presença dos atacantes.

![ResNet18 com FedMedian + norm filtering + clipping](assets/resnet_with_defense.png)

---


