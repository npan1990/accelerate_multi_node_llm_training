# Distributed Training

This is a tutorial for running a simple training with Pipeline Parallelism and Tensor Parallelism using accelerate.
It trains a small LLM on 2 Nodes with 1 GPU.

# How can I run this?

The first step is to start the containers. Each containers contains one or more GPUs. 
If a container contains more GPU change the hosts to include more slots. 

```bash
docker compose up --build
```

Connect to the two containers.

```bash
docker exec -it distributed_neural_nets_node1_1 bash
ssh node2
```

```bash
docker exec -it distributed_neural_nets_node1_2 bash
ssh node1
```

On node1 run:

```bash
accelerate launch --config_file default_config1 scripts/training.py
```

One node2 run:

```bash
accelerate launch --config_file default_config2 scripts/training.py
```