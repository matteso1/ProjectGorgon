
---

# Project Gorgon: High-Performance Speculative Decoding Engine

**Target Outcome:** A production-grade inference engine achieving >2x speedup on Llama-3-8B using custom CUDA kernels and Medusa-style speculative decoding.

## I. The Architecture Overview

Before writing code, understand what you are building.

1. **The Backbone:** A frozen, 4-bit quantized **Llama-3-8B** model. It acts as the "verifier."
2. **The Medusa Heads:** Small, trained MLP layers that sit on top of the backbone.
* *Head 1* predicts token 
* *Head 2* predicts token 
* *Head 3* predicts token 


3. **The Tree Attention Kernel:** A custom Triton kernel that allows the Backbone to verify a non-linear "tree" of candidate tokens in a single parallel forward pass.
4. **The Manager:** A Python control loop that manages the KV-cache, constructs the tree, and accepts/rejects tokens.

---

## II. Phase 1: The "Draft Heads" (Weeks 1-4)

**Goal:** Train the lightweight heads to predict future tokens.

### 1.1 The Theoretical Foundation

* **Read:** [Medusa Paper](https://arxiv.org/abs/2401.10774) - Specifically **Section 3.1**.
* **Concept:** You are NOT training a separate draft model. You are training simple residual blocks that take the *last hidden state* of Llama-3 and project it to the vocabulary size.
* **Architecture:**
```python
class MedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size)
        )
        # Crucial: Initialize weights to zero so it starts as a "pass-through"

```



### 1.2 The Implementation Steps

#### Step A: Data Prep

You need a dataset of text. Use **ShareGPT** or **RedPajama**.

* **Action:** Create a PyTorch `Dataset` class.
* **Logic:** For a sequence `[A, B, C, D, E]`:
* Input to Model: `[A, B, C, D]`
* Target for *Head 1*: `[B, C, D, E]` (Offset +1)
* Target for *Head 2*: `[C, D, E, _]` (Offset +2)
* Target for *Head 3*: `[D, E, _, _]` (Offset +3)



#### Step B: The Training Loop (The "Surgery")

You cannot load Llama-3 in full precision (it requires ~16GB VRAM). You must use `bitsandbytes` to load it in 4-bit (NF4).

* **Challenge:** You cannot backpropagate through a 4-bit frozen model easily.
* **Solution:** You don't need to! You only backpropagate through the *Heads*. The Llama-3 backbone is just a "feature extractor."
* **Prompt for LLM Helper:**
> "Help me write a PyTorch training script. I want to load `Llama-3-8B-Instruct` using `bitsandbytes` in 4-bit quantization. Then, attach 4 `MedusaHead` modules to it. I want to freeze the Llama-3 parameters completely. I need a training loop that computes the CrossEntropyLoss for each head against its specific time-shifted target and sums the losses."



### 1.3 Validation Milestone

* **Do not proceed** until you can run a script that inputs: `"The capital of France is"`
* And your heads output:
* Head 1: `"Paris"` (High confidence)
* Head 2: `"."` (High confidence)
* Head 3: `"\n"` (Lower confidence)



---

## III. Phase 2: The "Tree Attention" Kernel (Weeks 5-8)

**Goal:** Verify 64+ candidates in one GPU pass. This is the **"Holy Shit"** part of the resume.

### 2.1 The Problem

Standard Attention (`torch.nn.functional.scaled_dot_product_attention`) assumes a causal mask: Token 5 attends to 1, 2, 3, 4.
In Speculative Decoding, you verify a **Tree**.

* Branch A: `The -> cat -> sat`
* Branch B: `The -> dog -> ran`
* *Token "sat" (Branch A)* must attend to `The`, `cat`.
* *Token "ran" (Branch B)* must attend to `The`, `dog`.
* *Token "ran"* **MUST NOT** attend to `cat`.

### 2.2 The Solution: Custom Triton Kernel

You will use **OpenAI Triton** to write a kernel that accepts a custom "Topology Mask."

* **Read:** [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
* **Read:** [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) (Just the concepts of tiling).

### 2.3 The Implementation Steps

#### Step A: Flattening the Tree

You cannot pass a "Tree object" to a GPU. You must "flatten" it.

* **Tree:**
```text
      /-- B (Node 1)
A (Root)
      \-- C (Node 2)

```


* **Flat Tensor:** `[A, B, C]`
* **Parent Index:** `[-1, 0, 0]` (B's parent is index 0, C's parent is index 0).

#### Step B: The Kernel Logic

You need to modify the "Mask Loading" block of a standard attention kernel.

* **Standard Mask:** `if row < col: mask = -inf`
* **Your Tree Mask:** `if not is_ancestor(col, row): mask = -inf` (Conceptually).
* **Optimization:** Pre-compute the `TreeMask` tensor on CPU/Python and pass it to the kernel.
* **Prompt for LLM Helper:**
> "I need to write a Triton kernel for 'Tree Attention'. I have a tensor of flattened tokens `Q, K, V` and a boolean mask matrix `TreeMask` of shape `[Batch*TreeSize, Batch*TreeSize]`. Help me modify the standard Triton FlashAttention tutorial code to load this custom mask from memory instead of computing the causal mask on the fly."



### 2.4 Validation Milestone

* Create a unit test. Define a small tree. Run your kernel. Compare the output manually against a naive PyTorch implementation using a for-loop over branches. They must match, but yours should be faster.

---

## IV. Phase 3: The Inference Engine (Weeks 9-10)

**Goal:** Stitch the Heads (Phase 1) and the Kernel (Phase 2) into a generation loop.

### 3.1 The "Gorgon" Loop

1. **Draft:** Run the `Heads` on the current token. Get top-k predictions for each head.
2. **Tree Construction:** Build the candidate tree (Cartesian product of top-k choices).
3. **Verify:** Run the **Llama-3 Backbone** using your **Tree Attention Kernel** on all candidates at once.
4. **Accept:** Compare the Backbone's logits with the Draft's choices.
* *Greedy decoding:* If `Backbone(x) == Draft(x)`, accept.
* *Stochastic:* Use "Speculative Sampling" rejection criteria (Equation 2 in the Leviathan paper).


5. **KV Cache Update:** This is tricky. You generated KV-cache entries for *all* candidates (valid and invalid). You must **prune** the KV-cache to remove the "rejected" branches before the next step.

### 3.2 KV Cache Management

* **Naive:** Re-allocate cache every step (Slow).
* **Pro:** "Paged Attention" style (Too hard for 3 months).
* **Gorgon Approach:** "Cache Slicing." Maintain a flat KV cache. When you accept Branch A, copy Branch A's KV slots to the permanent position and overwrite the rest next time.

---

## V. Phase 4: The Interface & Visualization (Weeks 11-12)

**Goal:** The viral demo.

### 4.1 The Backend API

* Use **FastAPI**.
* Endpoint: `/generate_stream` (WebSocket).
* Payload: It sends the text chunk *plus* metadata:
```json
{
  "text": " The",
  "tree_debug": {
     "candidates": ["cat", "dog", "car"],
     "accepted": [true, false, false],
     "speedup": 2.4
  }
}

```



### 4.2 The Frontend

* **Tech:** React + D3.js (or `react-d3-tree`).
* **Visual:** A horizontal tree that grows to the right.
* Nodes flash **Green** instantly when accepted.
* Nodes turn **Red** and fade out when rejected.
* A live "Tokens Per Second" speedometer in the corner.



---

## VI. The Reading List (Annotated)

**1. The "Bible" (Read first):**

* **Title:** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*
* **Why:** This *is* your project. Read it until you understand Figure 2 (Tree Attention).
* C:\Users\nilsm\Desktop\VSCODE PROJECTS\ProjectGorgon\ProjectGorgon\literature\2401.10774v3 MEDUSA.pdf *

**2. The "Logic" (Read second):**

* **Title:** *Accelerating Large Language Model Decoding with Speculative Sampling* (Leviathan et al.)
* **Why:** Explains the math of *why* this is mathematically identical to standard generation (lossless).
*  C:\Users\nilsm\Desktop\VSCODE PROJECTS\ProjectGorgon\ProjectGorgon\literature\2302.01318v1 Accelerating Large Language Model Decoding.pdf *

**3. The "Tool" (Read when stuck on Triton):**

* **Title:** *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*
* **Why:** If you get stuck on "Why is my kernel segfaulting?", skim the architecture sections here.
* (https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) *
* C:\Users\nilsm\Desktop\VSCODE PROJECTS\ProjectGorgon\ProjectGorgon\literature\2019-mapl-tillet-kung-cox Triton.pdf *
  

**4. The "Optimization" (Optional):**

* **Title:** *FlashAttention-2*
* **Why:** To understand memory coalescing. If your kernel is slow, this paper tells you why.
* C:\Users\nilsm\Desktop\VSCODE PROJECTS\ProjectGorgon\ProjectGorgon\literature\2307.08691v1 Faster Attention with Better Parallelism and Work Partitioning.pdf *

---

## How to Start TODAY

1. **Set up the Repo:** `git init project-gorgon`.
2. **Environment:** Install `torch`, `triton`, `bitsandbytes`, `transformers`.
3. **First Commit:** Write a script that loads Llama-3-8B in 4-bit and prints "Hello World".
4. **Start Phase 1.**