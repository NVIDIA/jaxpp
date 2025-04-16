## Secrets to achieve high performance

Preliminary reading about transformers and parallelization can be found in [Jax's scaling book](https://jax-ml.github.io/scaling-book/).
If in a rush, read at least [Parallelization](https://jax-ml.github.io/scaling-book/training/) (lacks details for PP but see below).
Additional explanations and visualizations can be found in [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).

For details on B200 and other XLA flags see [here](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/GPU_performance.md#tips-for-good-llm-training-performance-on-blackwell-b200).

### Pipeline Parallelism details
The goal of a llm training job is to maximize throughput, tokens/s.
The key idea is to make TP and PP as small as possible, enough to
fit the model with activations in memory.
PP needs to be small to avoid a long pipeline, leading to more bubbles.

Then scale DP since it scales nearly perfectly, doubling the throughput as we double the GPUs.

A training step is

   $\theta_\text{DP}'$, $\mu\theta_\text{DP}'$, $\nu\theta_\text{DP}'$ = train_step($\theta_\text{DP}$, $\mu\theta_\text{DP}$, $\nu\theta_\text{DP}$, $X$)
   1. $\theta_\text{DP}$ is the model parameters
   2. $\mu\theta_\text{DP}$ and $\nu\theta_\text{DP}$ are the optimizer state if using Adam
      (some optimizers might have less, or no parameters).
      - At large DP with sharded optimizer, these become in the order of MiB (per DP rank)
      - For small DP, to gain more memory one can **offload** this optimizer state
   3. $X$ is the batch

This can be decomposed in three phases of the computation when using pipeline parallelism (PP):

1. Before gradient accumulation (GA) loop:
   
   $\hat\theta$, $\nabla\hat\theta$ = before_loop($\theta_\text{DP}$)

   "Master" (fp32) model parameters $\theta_\text{DP}$ are casted to bf16
   (let's call the casted version $\hat\theta_\text{DP}$),
   _then_ `all-gather`ed across data-parallel (DP) ranks
   to produce $\hat\theta$ which are replicated across DP and will be used in the
   GA loop.

   Additionally initial "zero gradients" $\nabla\hat\theta$ are allocated of size $\hat\theta$
   (usually GA is done in bf16), which will be used in the GA loop to accumulate
   the gradients into. These are replicated just like $\hat\theta$

2. GA loop

   $\nabla\hat\theta$ = ga_loop($\hat\theta$, $\nabla\hat\theta$, $X$, GA)

   there's no DP communication.
   The amount of memory required here other than the "live" state
   is just enough to keep the activations of a microbatch (of size $X$ / GA).
   Activations are a large part of memory usage.
   These is the part where **schedules**, **rematerialization**, and (activation) **offloading**
   (of older microbatches) can help.

   You can tradeoff memory with more rematerialization at the cost of more computation.

   The loop is the most computation heavy part (90%).
   Here you'd want to optimize TP/EP/CP/SP sharding rules to minimize such collectives
   since they are run at each loop iteration.

   This also means that you'd want to reduce GA (loop iterations) to a minimum, maybe
   at the cost of a higher bubble which you can recover from with better schedules like ZB.

3. After GA loop:

   $\theta_\text{DP}'$, $\mu\theta_\text{DP}'$, $\nu\theta_\text{DP}'$ = 
after_loop($\theta_\text{DP}$, $\nabla\hat\theta$, $\mu\theta_\text{DP}$, $\nu\theta_\text{DP}$)

   All DP-ranks _`reduce-scatter`_ their gradients $\nabla\hat\theta$ to "synchronize them".
   It's a `reduce-scatter` instead of an `all-reduce` because they need only $\nabla\hat\theta_\text{DP}$
   to update their corresponding $\hat\theta_\text{DP}$.

