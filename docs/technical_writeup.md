# Technical Deep Dive: Speculative Decoding with Tree Attention

## The Problem

Autoregressive LLM inference is fundamentally **memory-bandwidth bound**. Each token requires a full forward pass through the model, but most of the computation is just moving weights from VRAM to compute units. The GPU ALUs sit idle waiting for data. This means generating 128 tokens takes 128 sequential forward passes — even though the GPU could easily handle more work per pass.

**Speculative decoding** breaks this bottleneck: use a cheap model to *draft* multiple tokens, then *verify* them all in a single backbone pass. If the drafts are good enough, you get multiple tokens for the price of one forward pass.

## Architecture

### Medusa Heads

Instead of a separate draft model, Medusa attaches lightweight "heads" directly to the backbone's hidden states. Each head is a simple 2-layer MLP:

```
h_t → Linear(hidden_dim, hidden_dim) → SiLU → Linear(hidden_dim, vocab) → logits
```

Head *k* is trained to predict the token at position `t + k`. This is much cheaper than running a separate draft model because:

1. We reuse the backbone's hidden state (no extra computation)
2. Each head is ~2M parameters vs the backbone's 8B
3. Heads share the same representation space as the backbone

### Tree-Structured Candidates

Given 4 heads with top-k=4, we build a candidate tree:

```
         root
        / | | \
      h1₁ h1₂ h1₃ h1₄        (4 candidates from head 1)
      /||\  /||\  /||\  /||\
     h2   h2   h2   h2         (4 × 4 = 16 from head 2)
     ...
```

Total candidates: `4 + 16 + 64 + 256 = 340` across 4 levels.

Each root-to-leaf path represents a possible continuation sequence. We verify **all** of them in a single forward pass.

### Tree Attention

Standard causal attention uses a triangular mask. Tree attention generalizes this — each candidate token attends only to its **ancestors** in the tree (and itself):

```
mask[i, j] = True  iff  j is an ancestor of i (or i == j)
```

This maintains causal consistency: each candidate only sees the tokens it would have seen if generated sequentially along its path.

We implement this as a fused GPU kernel in both **Triton** and **CUDA**:

**Triton kernel** (`tree_attention_triton.py`):

- Per-row parallelism: one program per query position
- Tiled score computation with `BLOCK_D` for memory efficiency
- Masked softmax with `-inf` for disallowed positions
- Parameterized `BLOCK_N` and `BLOCK_D` as `tl.constexpr`

**CUDA kernel** (`tree_attention.cu`):

- One block per row, threads handle columns
- Shared memory for attention scores
- Serial softmax normalization per row
- Falls back gracefully for n, d ≤ 1024

### Verification & Acceptance

After the tree-attention forward pass, we check each root-to-leaf path:

1. For each path, compare the verifier's argmax at each position with the draft token
2. Accept the **longest matching prefix** across all paths
3. Collect the verifier's own prediction at the last accepted position as a **bonus token** (free!)

This means even if all drafts are wrong, we still get 1 token (the bonus) — the same as autoregressive decoding. If drafts are right, we get `len(accepted) + 1` tokens per iteration.

### KV-Cache Management

Speculative decoding complicates KV-cache management:

- Before drafting: **checkpoint** the cache
- After verification: **trim** to keep only accepted positions
- On full rejection: **rollback** to checkpoint

The `GorgonKVCache` class wraps HuggingFace-style `past_key_values` with these operations.

## Training the Heads

Medusa heads are trained via **knowledge distillation** from the frozen backbone:

1. Run backbone on training text → get hidden states + next-token logits
2. Head *k* receives hidden state at position *t*, must predict token at position `t + k`
3. Loss = cross-entropy between head logits and shifted ground truth
4. Only head parameters are updated; backbone stays frozen

Training is extremely efficient:

- The backbone forward pass is done once with `torch.no_grad()`
- Only the head parameters (~8M total for 4 heads) require gradients
- Training on 500 RedPajama samples takes ~30 minutes on RTX 4090

## Performance Analysis

The theoretical speedup of speculative decoding is:

\[
\text{speedup} = \frac{\text{avg tokens per iteration}}{1 + \text{overhead ratio}}
\]

Where:

- **avg tokens per iteration** = `E[accepted] + 1` (including bonus token)
- **overhead ratio** = cost of tree verification / cost of single autoregressive step

For tree-structured Medusa with good head accuracy:

- 60% acceptance rate → ~2.5 tokens per iteration
- Tree attention overhead is minimal (fused kernel, same FLOP budget)
- Expected speedup: **2-2.5×**

The key insight is that tree verification is only slightly more expensive than a single forward pass (same number of parameters touched), but yields multiple tokens.
