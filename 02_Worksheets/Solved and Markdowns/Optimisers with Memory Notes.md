# Deep Learning — Worksheet 06
# Optimisers With Memory

> **Big idea:** A gradient step is useful; a remembered gradient can be better.

You already know gradient descent:

$$\theta_{t+1}=\theta_t-\alpha g_t$$

where:

- `θ` = parameter / weight
- `g` = gradient
- `α` = learning rate

The new idea in this worksheet is simple:

> **Plain SGD reacts only to the current gradient. Memory-based optimisers remember useful information from earlier steps.**

## The whole story

```text
SGD → EWMA → Momentum → NAG
                 ↓
AdaGrad → RMSProp → Adam → AdamW
```

Easy meaning:

```text
SGD      → reacts
EWMA     → remembers
Momentum → uses remembered direction
NAG      → checks ahead
AdaGrad  → counts gradient scale
RMSProp  → forgets old scale
Adam     → combines direction + scale
AdamW    → Adam + separate weight decay
```

---

# 🎯 Contest Formula Cheat Sheet (Read This First!)

| Optimiser | Key Formula | What to Compute |
|---|---|---|
| **SGD** | $\theta_{t+1}=\theta_t-\alpha g_t$ | Subtract `α × gradient` |
| **EWMA** | $s_t=\beta s_{t-1}+(1-\beta)x_t$ | Keep `β` of old + `(1-β)` of new |
| **Momentum** | $v_t=\beta v_{t-1}+(1-\beta)g_t$, then $\theta_{t+1}=\theta_t-\alpha v_t$ | Update velocity, then move |
| **NAG** | $\theta_{look}=\theta_t-\alpha\beta v_t$, then compute $g_{look}$, then $v_{t+1}$, then $\theta_{t+1}$ | Predict → Measure → Correct → Move |
| **AdaGrad** | $r_t=r_{t-1}+g_t^2$, then $\theta_{t+1}=\theta_t-\alpha\frac{g_t}{\sqrt{r_t}}$ | Accumulate $g^2$, divide gradient by $\sqrt{r}$ |
| **RMSProp** | $s_t=\rho s_{t-1}+(1-\rho)g_t^2$, then $\theta_{t+1}=\theta_t-\alpha\frac{g_t}{\sqrt{s_t}}$ | Fading $g^2$, divide gradient by $\sqrt{s}$ |
| **Adam** | $m_t$, $v_t$, bias correct $\hat{m}_t$, $\hat{v}_t$, then $\theta_{t+1}=\theta_t-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}$ | Two memories + bias correction |

> **Default PyTorch Adam values**: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\alpha=0.001$

---

# 1. SGD — the baseline

## What is a gradient?

A gradient tells us the direction in which the loss increases.

Gradient descent wants to go **opposite** to that direction:

$$\boxed{\theta_{t+1}=\theta_t-\alpha g_t}$$

So:

```text
gradient         → uphill direction
gradient descent → downhill direction
```

The learning rate `α` tells us how far to move.

- gradient = **direction**
- learning rate = **distance**

## Why can SGD be noisy?

With stochastic or mini-batch training, `g_t` is only an estimate from a small sample.

One batch may point:

- slightly uphill
- sideways
- or even opposite to the useful long-term direction

### Simple analogy

SGD is like steering a boat by looking only at the **newest ripple**.

That is why later optimisers add memory.

---

# 2. EWMA — the first memory

## What is EWMA?

EWMA means **Exponentially Weighted Moving Average**.

It keeps a fading summary of recent observations:

$$\boxed{s_t=\beta s_{t-1}+(1-\beta)x_t}$$

where:

- `s_t` = current memory
- `s_{t-1}` = old memory
- `x_t` = newest value
- `β` = memory strength

### What does β do?

- large `β` → longer memory, slower reaction
- small `β` → shorter memory, faster reaction

The newest observation gets coefficient:

$$1-\beta$$

Old observations keep getting multiplied by `β`, so their influence fades.

### Coefficient of older values (Exam Question!)

Expanding the EWMA formula:

$$s_t = (1-\beta)x_t + \beta(1-\beta)x_{t-1} + \beta^2(1-\beta)x_{t-2} + \ldots$$

So the coefficient of `x_{t-k}` is $\beta^k(1-\beta)$. Older = smaller coefficient.

### Simple analogy

Imagine a coach judging a player's form.

Yesterday matters a lot.
Last week matters less.
Last month matters even less.

The coach does not keep every old observation separately. One running impression is enough.

---

## EWMA example

Given:

$$s_0=0,\quad \beta=0.8,\quad x_1=5,\quad x_2=1$$

So:

$$s_t=0.8s_{t-1}+0.2x_t$$

First update:

$$s_1=0.8(0)+0.2(5)=\boxed{1}$$

Second update:

$$s_2=0.8(1)+0.2(1)=\boxed{1}$$

The important idea is not the number `1`; it is that **the state carries memory from the previous step**.

If we expand the rule:

$$s_t=\beta^2s_{t-2}+\beta(1-\beta)x_{t-1}+(1-\beta)x_t$$

So older observations have smaller coefficients.

> **EWMA stores one running state, not the full history.**

---

# 3. Momentum — remembered direction

## What problem does Momentum solve?

Mini-batch gradients can zig-zag in a ravine:

```text
The useful direction (along the valley floor): →→→→→
SGD actual path:                               ↗ ↘ ↗ ↘ ↗ ↘
```

Even though the useful overall direction is consistent, SGD oscillates wildly sideways.

Momentum smooths that behaviour by keeping a **velocity**:

$$\boxed{v_t=\beta v_{t-1}+(1-\beta)g_t}$$

Then move using the velocity:

$$\boxed{\theta_{t+1}=\theta_t-\alpha v_t}$$

### Easy meaning

> **Momentum remembers where recent gradients have been pointing.**

Think about a rolling bicycle: one small wobble does not instantly reverse its whole direction.

---

## Momentum example

Given:

$$v_0=0,\quad \theta_0=2.00,\quad \beta=0.9,\quad \alpha=0.1$$

and:

$$g_1=4,\quad g_2=-2$$

### First velocity

$$v_1=0.9(0)+0.1(4)=\boxed{0.4}$$

Update:

$$\theta_1=2.00-0.1(0.4)=\boxed{1.96}$$

### Second velocity

$$v_2=0.9(0.4)+0.1(-2)$$

$$=0.36-0.2=\boxed{0.16}$$

Update:

$$\theta_2=1.96-0.1(0.16)=\boxed{1.944}$$

Notice:

$$g_2=-2$$

but:

$$v_2=+0.16$$

So the parameter **still moves in the same overall direction** because Momentum remembers the previous direction.

> **Momentum remembers a direction, not a list of old gradients.**

---

# 4. NAG — look ahead

NAG = **Nesterov Accelerated Gradient**.

Momentum uses remembered direction.

NAG adds:

> **Look ahead before measuring the new gradient.**

## The three main equations

First predict (look-ahead position):

$$\boxed{\theta_{look}=\theta_t-\alpha\beta v_t}$$

Measure the gradient at that future-looking point:

$$\boxed{g_{look}=\nabla L(\theta_{look})}$$

Correct the velocity:

$$\boxed{v_{t+1}=\beta v_t+(1-\beta)g_{look}}$$

Finally move:

$$\boxed{\theta_{t+1}=\theta_t-\alpha v_{t+1}}$$

### Order matters

```text
1. Predict   → θ_look = θ - α·β·v
2. Measure   → compute gradient at θ_look
3. Correct   → update velocity with the look-ahead gradient
4. Move      → θ_{t+1} = θ - α·v_{t+1}
```

### Easy analogy

A cyclist approaching a bend looks where the bicycle is **about to be**, not only where it is now.

So:

$$\boxed{\text{Momentum: what is the slope here?}}$$

$$\boxed{\text{NAG: what is the slope where momentum is taking me?}}$$

---

## NAG example

Given:

$$\theta_t=2.000,\quad v_t=0.500,\quad \alpha=0.1,\quad \beta=0.9$$

and:

$$\nabla L(\theta_{look})=3.000$$

### Step 1 — Look ahead

$$\theta_{look}=2.000-0.1(0.9)(0.500)$$

$$\boxed{\theta_{look}=1.955}$$

### Step 2 — Velocity contributions

$$0.9(0.500)=0.45$$

$$0.1(3.000)=0.3$$

### Step 3 — Corrected velocity

$$v_{t+1}=0.45+0.3=\boxed{0.75}$$

### Step 4 — Real update

$$\theta_{t+1}=2.000-0.1(0.75)$$

$$\boxed{1.925}$$

---

# 5. AdaGrad — different parameters need different step sizes

Momentum is mainly about **direction**.

AdaGrad asks:

> **How large should the step be for this parameter?**

Different coordinates may get very different gradient sizes.

## Formula

AdaGrad stores cumulative squared gradients:

$$\boxed{r_t=r_{t-1}+g_t^2}$$

Then:

$$\boxed{\theta_{t+1}=\theta_t-\alpha\frac{g_t}{\sqrt{r_t}+\epsilon}}$$

The worksheet often tells you to ignore `ε` for hand calculations.

### Why square the gradient?

Squaring removes its sign when measuring scale.

But the original `g_t` stays in the numerator, so the update still has the correct direction.

### How it adapts

- Parameter with **large, frequent gradients** → large $r_t$ → small effective step
- Parameter with **small, infrequent gradients** → small $r_t$ → large effective step

> AdaGrad **automatically reduces the learning rate for "busy" parameters** and keeps it large for "quiet" ones.

---

## AdaGrad example

Given:

$$r_{t-1}=(0,0),\quad g_t=(4,1)$$

### Square

$$g_t^2=(16,1)$$

### Accumulate

$$r_t=(16,1)$$

### Square root

$$\sqrt{r_t}=(4,1)$$

### Normalise

$$\frac{g_t}{\sqrt{r_t}}=\left(\frac44,\frac11\right)=\boxed{(1,1)}$$

### AdaGrad's problem

The accumulator only grows:

$$r_t=r_{t-1}+g_t^2$$

It **never forgets**.

Eventually the denominator becomes very large, making learning extremely slow (learning rate effectively → 0).

That motivates RMSProp.

---

# 6. RMSProp — forget old scale

RMSProp keeps the good idea of AdaGrad but gives the squared-gradient memory a **fade**.

## Formula

$$\boxed{s_t=\rho s_{t-1}+(1-\rho)g_t^2}$$

and:

$$\boxed{\theta_{t+1}=\theta_t-\alpha\frac{g_t}{\sqrt{s_t}+\epsilon}}$$

### AdaGrad vs RMSProp

```text
AdaGrad  → r_t = r_{t-1} + g²        (accumulates forever, never forgets)
RMSProp  → s_t = ρ·s_{t-1} + (1-ρ)g² (EWMA of g², recent matters more)
```

### Easy analogy

AdaGrad is like keeping every old weather report forever.

RMSProp remembers the **recent climate** instead.

---

## RMSProp example

Given:

$$s_{t-1}=(16,1),\quad g_t=(4,1),\quad \rho=0.9$$

### Square the new gradient

$$g_t^2=(16,1)$$

### Keep 90% of old state

$$0.9(16,1)=(14.4,0.9)$$

### Add 10% of new squared gradient

$$0.1(16,1)=(1.6,0.1)$$

### New state

$$s_t=(14.4,0.9)+(1.6,0.1)$$

$$\boxed{s_t=(16,1)}$$

It happens to equal the old state in this example.

But RMSProp is **not** the same as AdaGrad because old information will keep getting discounted in later steps.

---

# 7. Adam — two memories

Adam = **Adaptive Moment Estimation**.

Adam combines:

1. **1st moment (Momentum-like direction memory)** — *where to go*
2. **2nd moment (RMSProp-like scale memory)** — *how big a step*

So Adam answers:

> **Which way should I move?**

and:

> **How large should the step be?**

---

## 7.1 First memory: direction (1st moment)

$$\boxed{m_t=\beta_1m_{t-1}+(1-\beta_1)g_t}$$

`m_t` is the signed direction memory. Think of it as a smoothed gradient.

---

## 7.2 Second memory: scale (2nd moment)

$$\boxed{v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2}$$

`v_t` stores gradient scale (how large gradients have been).

Important naming (often confused!):

$$\boxed{m_t=\text{direction/sign memory (1st moment)}}$$

$$\boxed{v_t=\text{scale memory (2nd moment)}}$$

---

# 8. Adam bias correction

At the beginning:

$$m_0=0,\quad v_0=0$$

This creates an artificial early problem: the moving averages look too small because they start from zero.

Adam corrects this:

$$\boxed{\hat m_t=\frac{m_t}{1-\beta_1^t}}$$

$$\boxed{\hat v_t=\frac{v_t}{1-\beta_2^t}}$$

The hats mean **bias-corrected estimates**.

### Why does this work?

At step $t=1$ with $\beta_1=0.9$:
- $m_1 = 0.9(0) + 0.1 g_1 = 0.1 g_1$ ← looks tiny
- $\hat{m}_1 = \frac{0.1 g_1}{1 - 0.9^1} = \frac{0.1 g_1}{0.1} = g_1$ ← corrected back to true value!

As $t \to \infty$, $1-\beta^t \to 1$, so bias correction becomes negligible.

### Simple analogy

Imagine a class average starts with several imaginary zero marks.

The first real mark looks too small.

Bias correction removes the effect of those imaginary zeros.

---

# 9. Adam update rule

After correction:

$$\boxed{\theta_{t+1}=\theta_t-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}}$$

So:

- `m̂` → remembered direction (normalised gradient direction)
- `v̂` → adaptive scale (how volatile this parameter has been)
- `α` → overall learning-rate control

---

## Adam example

Given:

$$m_0=0,\quad v_0=0$$

$$g_1=4$$

$$\beta_1=0.9,\quad \beta_2=0.999,\quad \alpha=0.01$$

Ignore `ε`.

**First moment:**

$$m_1=0.9(0)+0.1(4)=\boxed{0.4}$$

**Second moment:**

$$v_1=0.999(0)+0.001(16)=\boxed{0.016}$$

**Bias-correct first moment** ($t=1$):

$$\hat m_1=\frac{0.4}{1-0.9^1}=\frac{0.4}{0.1}=\boxed{4}$$

**Bias-correct second moment** ($t=1$):

$$\hat v_1=\frac{0.016}{1-0.999^1}=\frac{0.016}{0.001}=\boxed{16}$$

**Normalised direction:**

$$\frac{\hat m_1}{\sqrt{\hat v_1}}=\frac{4}{\sqrt{16}}=\frac{4}{4}=\boxed{1}$$

**Movement magnitude:**

$$\Delta\theta_1=0.01(1)=\boxed{0.01}$$

So Adam uses both **direction memory and scale memory**.

---

# 10. AdamW — Adam + separate weight decay

Adam decides the adaptive training direction.

Weight decay has a different job:

> **Prefer smaller parameter values.**

AdamW keeps these jobs separate.

## Formula

$$\boxed{\theta_{t+1}=(1-\alpha\lambda)\theta_t-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}}$$

There are two separate pieces.

### Weight decay

$$\boxed{(1-\alpha\lambda)\theta_t}$$

This shrinks the parameter toward zero before the Adam update.

### Adam move

$$\boxed{\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}}$$

This follows Adam's adaptive direction.

### Easy analogy

Steering a bicycle toward the destination and keeping the bicycle close to the centre of a lane are two separate controls.

AdamW keeps these two jobs separate.

> **Note**: Regular Adam folds weight decay *into* the gradient, which interferes with the adaptive scaling. AdamW applies it *directly* to the parameter, which is more correct.

---

# 11. AdamW example

Given:

$$\theta_t=1.00$$

$$\alpha=0.01,\quad \lambda=0.1$$

and normalised Adam direction:

$$\frac{\hat m_t}{\sqrt{\hat v_t}} = 0.50$$

### Decay

$$(1-0.01 \times 0.1)(1.00) = (1 - 0.001)(1.00) = \boxed{0.999}$$

### Adaptive move

$$0.01 \times 0.50 = \boxed{0.005}$$

### Combine

$$\theta_{t+1}=0.999-0.005$$

$$\boxed{\theta_{t+1}=0.994}$$

If the normalised Adam direction were negative, the adaptive term would increase `θ`.

But the decay term would still pull `θ` toward zero regardless.

---

# 12. Vanishing & Exploding Gradients (Syllabus!)

These are two opposite failure modes that happen during backpropagation through many layers.

## Vanishing Gradients

During backprop, each layer multiplies the gradient by its local derivative.

If derivatives are small (e.g. Sigmoid: max = 0.25):

$$\text{Gradient at layer 1} \approx 0.25 \times 0.25 \times 0.25 \times \ldots \approx 0$$

The gradient **shrinks to zero exponentially** as it travels to earlier layers.

**Effect**: Early layers stop learning entirely.

**Cause**: Sigmoid/Tanh activations in deep networks.

**Fix**: Use **ReLU** activations (gradient = 1 for positive inputs).

## Exploding Gradients

If weights are large or derivatives multiply to values > 1:

$$\text{Gradient at layer 1} \approx 2.0 \times 2.0 \times 2.0 \times \ldots \to \infty$$

The gradient **grows to infinity** — NaN loss, unstable training.

**Effect**: Loss becomes NaN, weights blow up to ±∞.

**Cause**: Large initial weights in deep networks.

**Fix**: **Gradient Clipping** — if $\|g\| > \text{threshold}$, scale it down:

$$g \leftarrow g \times \frac{\text{threshold}}{\|g\|}$$

In PyTorch:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Quick Comparison

| | Vanishing | Exploding |
|---|---|---|
| **Symptom** | Early layers never update | Loss = NaN, weights → ∞ |
| **Gradient size** | → 0 | → ∞ |
| **Main cause** | Sigmoid/Tanh activations | Large weights |
| **Fix** | ReLU, Batch Norm, ResNets | Gradient clipping |

---

# 13. PyTorch Quick Reference

```python
import torch.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# NAG (Nesterov)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# RMSProp
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adam (default β1=0.9, β2=0.999)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam + separate weight decay)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

# 14. The big comparison

## Direction family

```text
SGD
 ↓
EWMA
 ↓
Momentum
 ↓
NAG
```

### SGD
Current gradient only.
> **Reacts**

### EWMA
Fading average.
> **Remembers**

### Momentum
Uses remembered gradients as a direction/velocity.
> **Uses direction**

### NAG
Uses momentum and checks a predicted future location.
> **Checks ahead**

---

## Scale family

```text
AdaGrad
 ↓
RMSProp
 ↓
Adam
 ↓
AdamW
```

### AdaGrad
Cumulative squared gradients.
> **Counts scale (but never forgets)**

### RMSProp
Fading average of squared gradients.
> **Forgets old scale**

### Adam
Combines direction memory + scale memory + bias correction.
> **Combines**

### AdamW
Adam + separate weight decay.
> **Combines and decays**

---

# 15. Easy table for revision

| Optimiser | Memory Type | Main idea | PyTorch |
|---|---|---|---|
| **SGD** | None | Current gradient only | `optim.SGD(lr)` |
| **EWMA** | Direction (fading) | Fading memory | — (concept only) |
| **Momentum** | Direction | Remembers direction | `optim.SGD(lr, momentum=0.9)` |
| **NAG** | Direction + look-ahead | Momentum + look ahead | `optim.SGD(lr, momentum=0.9, nesterov=True)` |
| **AdaGrad** | Scale (cumulative) | Accumulated squared-gradient scale | `optim.Adagrad(lr)` |
| **RMSProp** | Scale (fading) | Fading squared-gradient scale | `optim.RMSprop(lr)` |
| **Adam** | Direction + Scale | Direction + scale memories + bias correction | `optim.Adam(lr)` |
| **AdamW** | Direction + Scale + Decay | Adam + separate weight decay | `optim.AdamW(lr, weight_decay)` |

---

# 16. How to choose one

| Problem | Natural starting idea |
|---|---|
| Noisy direction | **Momentum or NAG** |
| Coordinates have different scales | **AdaGrad or RMSProp** |
| Want direction + adaptive scale | **Adam** |
| Want Adam + explicit weight decay | **AdamW** |
| NaN / unstable loss | **Gradient Clipping** |

These are **starting hypotheses**, not universal rules.

---

# 17. The final worksheet questions

## Q1. EWMA coefficient of `x_{t-2}`

Starting from:

$$s_t=\beta s_{t-1}+(1-\beta)x_t$$

Expanding further gives the coefficient of `x_{t-2}`:

$$\boxed{\beta^2(1-\beta)}$$

## Q2. Why does an opposing gradient not immediately reverse Momentum?

Because Momentum still contains the old velocity:

$$v_t=\beta v_{t-1}+(1-\beta)g_t$$

So one opposite gradient may not be enough to cancel the remembered direction.

$$\boxed{\text{Momentum has memory — one opposing gradient is diluted by } \beta}$$

## Q3. NAG's four actions

$$\boxed{\text{Predict → Measure → Correct → Move}}$$

## Q4. AdaGrad vs RMSProp

AdaGrad:

$$r_t=r_{t-1}+g_t^2$$

→ keeps accumulating old squared gradients. Learning rate → 0 over time.

RMSProp:

$$s_t=\rho s_{t-1}+(1-\rho)g_t^2$$

→ old squared gradients fade. Effective learning rate stays alive.

$$\boxed{\text{AdaGrad remembers all scale; RMSProp remembers recent scale}}$$

## Q5. Why is AdamW decay separate?

Adam's normalised update decides **how to learn**.

Weight decay decides **how strongly parameters should shrink toward zero**.

Applying decay inside the gradient (like regular Adam) interferes with the adaptive scaling. AdamW applies it directly to the parameter value instead.

$$\boxed{\text{AdamW separates adaptive learning from parameter shrinkage}}$$

## Q6. What are Vanishing Gradients and how do you fix them?

Vanishing gradients occur when the chain rule product of many small derivatives shrinks the gradient to near zero before it reaches early layers.

$$\boxed{\text{Fix: Use ReLU (gradient = 1 for } z > 0\text{) instead of Sigmoid/Tanh}}$$

## Q7. What are Exploding Gradients and how do you fix them?

Exploding gradients occur when large weights cause the chain rule product to grow exponentially, eventually causing NaN loss.

$$\boxed{\text{Fix: Gradient Clipping — cap }\|g\|\text{ at a threshold before the update}}$$

---

# 18. The one-page memory trick

```text
SGD
→ current gradient only

EWMA
→ fading memory of values

Momentum
→ direction memory (smoothed velocity)

NAG
→ direction + look ahead before measuring

AdaGrad
→ accumulated scale (never forgets → learning dies)

RMSProp
→ fading scale (EWMA of g²)

Adam
→ direction (m_t) + scale (v_t) + bias correction

AdamW
→ Adam + separate weight decay term

Vanishing Gradient → gradients die → Fix: ReLU
Exploding Gradient → gradients blow up → Fix: Gradient Clipping
```

## One sentence to remember everything

> **SGD reacts, EWMA remembers, Momentum uses direction, NAG looks ahead, AdaGrad counts scale but forgets nothing, RMSProp forgets old scale, Adam combines direction and scale with bias correction, AdamW adds separate decay, ReLU prevents vanishing, and clipping prevents exploding.**
