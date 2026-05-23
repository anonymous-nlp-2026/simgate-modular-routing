# Counterfactual Decomposition Framework: Formal Draft

> A diagnostic framework for identifying and quantifying the sources of routing signal in multi-modal LLM agents.

---

## Part 1: Formal Definitions

### 1.1 Episode-Level Counterfactual Labeling Protocol

**Setup.** Let $E$ be a sequential decision-making environment, $\Pi = \{\pi_{\text{det}}, \pi_{\text{int}}\}$ a set of two execution policies (deterministic/tool-use and internal/generative), and $\mathcal{E} = \{e_1, e_2, \ldots, e_N\}$ a set of episodes (tasks).

**Counterfactual Execution.** For each episode $e \in \mathcal{E}$, execute both policies to obtain:

$$S_{\text{det}}(e) = \text{Score}(e, \pi_{\text{det}}), \quad S_{\text{int}}(e) = \text{Score}(e, \pi_{\text{int}})$$

where $\text{Score}: \mathcal{E} \times \Pi \to \mathbb{Z}$ is the environment's evaluation metric (integer scores; death imposes penalty $s(\tau) = -P$, with $P = 100$ in ScienceWorld).

**Oracle Label Function (Three-Way):**

Each episode yields a death indicator $d(\tau) \in \{0, 1\}$ for each policy. The label $\ell \in \{\texttt{det-pref}, \texttt{int-pref}, \texttt{no-pref}\}$ is assigned by:

$$\ell(e) = \begin{cases} \texttt{det-pref} & \text{if } d_{\text{int}}=1 \wedge d_{\text{det}}=0 \quad \text{(death asymmetry)} \\ \texttt{int-pref} & \text{if } d_{\text{int}}=0 \wedge d_{\text{det}}=1 \quad \text{(death asymmetry)} \\ \texttt{no-pref} & \text{if } d_{\text{int}}=1 \wedge d_{\text{det}}=1 \quad \text{(both dead)} \\ \texttt{det-pref} & \text{if } d_{\text{int}}=d_{\text{det}}=0 \wedge S_{\text{det}}(e) > S_{\text{int}}(e) \\ \texttt{int-pref} & \text{if } d_{\text{int}}=d_{\text{det}}=0 \wedge S_{\text{int}}(e) > S_{\text{det}}(e) \\ \texttt{no-pref} & \text{if } d_{\text{int}}=d_{\text{det}}=0 \wedge S_{\text{det}}(e) = S_{\text{int}}(e) \quad \text{(tied alive)} \end{cases}$$

*Intuition:* The label function answers "which policy would have been better for this episode?" Death asymmetry dominates (78.9% of non-tie labels); ties yield no-preference rather than a default.

**Contrast with Step-Level Oracle Routing (e.g., ITP-R):**

Step-level oracles attempt to assign credit at each decision step $t$:

$$L_{\text{step}}(e, t) = \arg\max_{\pi} \mathbb{E}[R_t \mid \pi, s_t]$$

In environments with sparse terminal reward (e.g., ScienceWorld gives score only at episode end), this requires per-step credit assignment which degenerates:

$$\text{Var}[\hat{L}_{\text{step}}(e, t)] \to \infty \quad \text{as reward sparsity increases}$$

Our episode-level protocol avoids this entirely by operating on the final outcome.

*Evidence:* ScienceWorld provides only terminal scores; step-level approaches cannot attribute signal to intermediate steps without auxiliary reward shaping.

---

### 1.2 Death Avoidance Fraction (DAF)

**Definition: Death Indicator.** For any trajectory $\tau$, the death indicator is:

$$d(\tau) \in \{0, 1\}$$

When $d(\tau) = 1$, the environment imposes a penalty: $s(\tau) = -P$ (with $P = 100$ in ScienceWorld). Both policies can die: $d(\tau_{\text{int}})$ and $d(\tau_{\text{det}})$ are recorded independently for each episode.

**Episode Partitions (over all episodes):**

$$\mathcal{E}_{\text{nd}} = \{e \in \mathcal{E} : d(\tau_{\text{det}}) = 0 \wedge d(\tau_{\text{int}}) = 0\}$$

i.e., $\mathcal{E}_{\text{nd}}$ contains episodes where *both* strategies survive. Any episode where either policy dies is excluded from the non-death subset.

**Full Routing Signal (all episodes):**

$$\Delta_{\text{full}} = \frac{1}{N} \sum_{i=1}^{N} \left[ S_{\text{det}}(e_i) - S_{\text{int}}(e_i) \right]$$

where $N = |\mathcal{E}|$ is the total number of episodes.

**Non-Death Routing Signal (all non-death episodes):**

$$\Delta_{\text{nd}} = \frac{1}{|\mathcal{E}_{\text{nd}}|} \sum_{e \in \mathcal{E}_{\text{nd}}} \left[ S_{\text{det}}(e) - S_{\text{int}}(e) \right]$$

**Death Avoidance Fraction:**

$$\text{DAF} = 1 - \frac{\Delta_{\text{nd}}}{\Delta_{\text{full}}}$$

*Intuition:* DAF measures how much the overall routing signal diminishes when death episodes are removed. Computed over ALL episodes (not just the det-preferred subset). If $\text{DAF} \approx 1$, removing death episodes eliminates nearly all routing signal—meaning the routing advantage is essentially death avoidance. If $\text{DAF} \approx 0$, routing signal persists even without death episodes.

**Boundary Cases:**
- $\text{DAF} = 1$: All routing signal is death avoidance. $\Delta_{\text{nd}} = 0$.
- $\text{DAF} = 0$: Routing signal is fully preserved after removing death episodes. $\Delta_{\text{nd}} = \Delta_{\text{full}}$.
- $\text{DAF} > 1$: Possible if non-death episodes favor $\pi_{\text{int}}$ on average (negative $\Delta_{\text{nd}}$), meaning death avoidance is the *only* reason $\pi_{\text{det}}$ wins overall.

**Bootstrap Confidence Interval:**

$$\text{CI}_{95\%}(\text{DAF}) = \left[ \text{DAF}^*_{(0.025)}, \ \text{DAF}^*_{(0.975)} \right]$$

where $\text{DAF}^*_{(q)}$ is the $q$-th quantile of $B = 10000$ bootstrap resamples of $\mathcal{E}$ (resampling all $N$ episodes).

*Evidence:* ScienceWorld DAF = 0.904, 95% CI = [0.816, 0.988] (N=245 episodes). Pearson $r$(DAF, routing accuracy) = 0.9698 across task types.

---

### 1.3 Signal Concentration Index (SCI)

**Motivation.** DAF is specific to death-vs-non-death decomposition. We generalize to measure how concentrated routing signal is across episodes, regardless of failure mode.

**Per-Episode Routing Advantage:**

$$\delta(e) = S_{\text{det}}(e) - S_{\text{int}}(e), \quad \forall e \in \mathcal{D}$$

**Signal Concentration Index (Gini-based):**

$$\text{SCI} = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |\delta_i - \delta_j|}{2n \sum_{i=1}^{n} \delta_i}$$

where $n = |\mathcal{D}|$ and $\delta_i$ are sorted in ascending order.

*Intuition:* SCI ∈ [0, 1]. High SCI (→1) means routing signal is concentrated in a few extreme episodes (e.g., death episodes with $\delta = 1.0$ while others have $\delta \approx 0$). Low SCI (→0) means signal is uniformly distributed across episodes.

**Relationship to DAF:**

$$\text{High DAF} \implies \text{High SCI}$$

but not vice versa. High SCI could result from any type of extreme episodes, not just death. DAF identifies *what* the extreme episodes are; SCI measures *how extreme* the concentration is.

**Alternative: Top-$k$ Concentration Ratio:**

$$\text{CR}_k = \frac{\sum_{i=n-k+1}^{n} \delta_{(i)}}{\sum_{i=1}^{n} \delta_{(i)}}$$

where $\delta_{(i)}$ are order statistics. This directly answers "what fraction of total signal comes from the top $k$ episodes?"

*Evidence:* In ScienceWorld, the death episodes (which are also the top-$\delta$ episodes) account for 90.4% of total signal, confirming extreme concentration.

---

## Part 2: Applicability Conditions

### 2.1 Framework Applicability

The Counterfactual Decomposition Framework applies when **all** of the following hold:

| Condition | Formal Requirement | Rationale |
|-----------|-------------------|-----------|
| Multi-modal execution | $|\Pi| \geq 2$ distinct policies | Need at least two modes to route between |
| Episode-level evaluation | $\exists$ Score: $\mathcal{E} \times \Pi \to \mathbb{R}$ | Need measurable outcome per episode per policy |
| Counterfactual feasibility | Each episode can be executed under all $\pi \in \Pi$ | Need paired comparisons |
| Non-trivial routing signal | $\exists e: S_{\pi_1}(e) \neq S_{\pi_2}(e)$ | If all modes perform identically, nothing to decompose |

### 2.2 DAF Informativeness Conditions

DAF is a meaningful diagnostic metric when:

**Necessary Condition — Failure Asymmetry:**

$$\Pr(\text{Death}(e) \mid \pi_{\text{int}}) \gg \Pr(\text{Death}(e) \mid \pi_{\text{det}})$$

*Intuition:* If both modes fail at the same rate, death episodes cancel out in the counterfactual comparison, and DAF cannot isolate death avoidance as a signal source.

**Sufficient Conditions for High-Information DAF:**
1. One mode has substantially higher catastrophic failure rate
2. Non-death performance differences are small relative to death-induced differences
3. Sufficient episodes to estimate DAF with tight CI ($N \geq 50$ episodes recommended)

**When DAF is Uninformative:**
- $\Pr(\text{Death} \mid \pi_{\text{int}}) = \Pr(\text{Death} \mid \pi_{\text{det}})$: Symmetric failure → DAF ≈ fraction of death episodes in $\mathcal{E}$, not diagnostic
- No death episodes exist: DAF = 0 trivially, use SCI instead
- All episodes are death: DAF = 1 trivially (tautological)

### 2.3 Validated Scenarios

#### ScienceWorld (Primary)

| Property | Value |
|----------|-------|
| Environment type | Text-based interactive science tasks (30 task types) |
| Score metric | Normalized task completion [0, 1] |
| Death mechanism | Agent enters irrecoverable states → score = 0 |
| Failure asymmetry | $\pi_{\text{int}}$ deaths frequent; $\pi_{\text{det}}$ deaths rare (scripted, avoids catastrophe) |
| **DAF** | **90.4%** |
| **95% CI** | **[81.6%, 98.8%]** |
| **Pearson $r$(DAF, accuracy)** | **0.9698** |
| Implication | Routing signal is overwhelmingly death avoidance; complex router unnecessary |

*Evidence:* Sensitivity analysis confirms robustness: DAF > 80% across all performance thresholds $P \geq 50$; DAF > 87% across all death threshold variations.

#### ALFWorld (Binary Success/Fail)

| Property | Value |
|----------|-------|
| Environment type | Embodied household tasks (6 task types) |
| Score metric | Binary {0, 1} (success/fail) |
| Death mechanism | Failure = score 0, but both modes can fail |
| **DAF** | **1.0 (trivially)** |
| Implication | DAF metric is tautological in binary environments (all routing signal is fail-avoidance by definition). However, framework still reveals: 5/6 task types show $\pi_{\text{det}}$ advantage, confirming directional pattern |

*Insight:* In binary-outcome environments, use the framework's counterfactual labeling (Part 1.1) and task-type decomposition rather than DAF specifically.

#### MATH (No Death, Signal Reversal)

| Property | Value |
|----------|-------|
| Environment type | Mathematical reasoning (5 difficulty levels) |
| Score metric | Binary correctness per problem |
| Death mechanism | None (wrong answer ≠ catastrophic failure) |
| **DAF** | **Not applicable** (no death asymmetry) |
| Key finding | Signal reversal: $\pi_{\text{int}}$ better on easy problems, $\pi_{\text{det}}$ better on hard problems |
| Statistical signal | Directional trend $p = 0.079$ |
| Implication | Framework reveals difficulty-dependent routing structure even without DAF |

*Evidence:* Counterfactual decomposition by difficulty level shows monotonic trend in mode preference.

#### Llama-3.1-8B Cross-Model Replication

| Property | Value |
|----------|-------|
| Model | Llama-3.1-8B-Instruct (vs. GPT-4o primary) |
| Environment | ScienceWorld |
| Key finding | $\text{int\_preferred} = 0$ (deterministic always preferred), $\pi_{\text{det}}$ never dies |
| Implication | Pattern replicates: weaker model shows even stronger death-avoidance dominance |

*Evidence:* Cross-model evaluation confirms framework's diagnostic applies regardless of base LLM capability.

---

## Part 3: Differentiation from Existing Methods

### 3.1 vs. ITP-R (Intuitive Tool-use for Planning and Reasoning)

| Dimension | ITP-R | Counterfactual Decomposition |
|-----------|-------|------------------------------|
| **Purpose** | Routing method (decides which tool/mode to use) | Analysis framework (diagnoses routing signal sources) |
| **Granularity** | Step-level oracle routing | Episode-level counterfactual |
| **Reward requirement** | Per-step reward or surrogate | Terminal reward sufficient |
| **Sparse reward** | Degrades (credit assignment problem) | Unaffected (uses final score only) |
| **Output** | Routing decisions | Diagnostic metrics (DAF, SCI) |
| **When to use** | At inference time | Before training a router |

**Key Advantage:** ITP-R requires step-level reward attribution. In environments with only terminal reward (ScienceWorld: score given only at episode end), step-level credit assignment becomes:

$$\hat{R}_t = R_T \cdot \frac{\partial \log \pi(a_t|s_t)}{\partial \theta} \quad \text{(high variance, uninformative)}$$

Our framework sidesteps this entirely by comparing episode outcomes.

### 3.2 vs. ReTool / AgentMath / Chameleon (Learned Routing Methods)

| Dimension | Learned Routers | Counterfactual Decomposition |
|-----------|----------------|------------------------------|
| **Assumption** | Learnable routing signal exists | Diagnoses whether learnable signal exists |
| **Phase** | Training/inference | Pre-training diagnostic |
| **Failure mode** | Train router on death-avoidance signal → overfits to task-type binary | Identifies this failure mode before training |
| **Generalization** | Assumes episode-level generalization | Reveals when generalization is impossible (C2) |

**Diagnostic Value:** Before investing compute in training a router, apply our framework:

$$\text{If DAF} \approx 1: \quad \text{Simple rule suffices (avoid tasks where } \pi_{\text{int}} \text{ dies)}$$
$$\text{If DAF} \approx 0: \quad \text{Signal is nuanced → learned router may be justified}$$

*Evidence:* In ScienceWorld, DAF = 90.4% correctly predicts that a trained router cannot generalize at episode level (cross-split accuracy = 47.1% ≈ random), because the "signal" is just a task-type-level death/no-death binary.

### 3.3 Core Value Proposition

**Three contributions of the Counterfactual Decomposition Framework:**

1. **Episode-level counterfactual protocol** solves the sparse-reward credit assignment problem that plagues step-level oracle routing approaches. No reward shaping or per-step attribution needed.

2. **DAF as pre-training diagnostic** answers "where does routing signal come from?" before committing resources to router training. High DAF → signal is death avoidance → simple rule suffices → don't train a complex router.

3. **Generalizability** of the framework itself: applicable to any multi-modal agent system with $|\Pi| \geq 2$ and episode-level evaluation, independent of environment, LLM backbone, or task domain.

**The meta-insight:** The framework shifts the research question from "how to build a better router" to "what is the router actually learning?" — and in many cases, the answer ("death avoidance") renders complex routing methods unnecessary.

---

## Summary of Notation

| Symbol | Definition |
|--------|-----------|
| $E$ | Environment |
| $\Pi = \{\pi_{\text{det}}, \pi_{\text{int}}\}$ | Policy set (deterministic, internal) |
| $\mathcal{E}$ | Episode set |
| $S_\pi(e)$ | Score of episode $e$ under policy $\pi$ |
| $\ell(e)$ | Oracle label $\in \{\texttt{det-pref}, \texttt{int-pref}, \texttt{no-pref}\}$ for episode $e$ |
| $\mathcal{D}$ | Det-preferred episodes: $\{e : L(e) = \pi_{\text{det}}\}$ (used in SCI) |
| $d(\tau)$ | Death indicator: $d(\tau) \in \{0,1\}$; $d=1 \Rightarrow s(\tau) = -P$ |
| $\mathcal{E}_{\text{nd}}$ | Non-death episodes: $\{e : d(\tau_{\text{det}})=0 \wedge d(\tau_{\text{int}})=0\}$ (both survive) |
| $\Delta_{\text{full}}$ | Mean routing advantage over all $N$ episodes |
| $\Delta_{\text{nd}}$ | Mean routing advantage over non-death episodes |
| DAF | Death Avoidance Fraction |
| SCI | Signal Concentration Index |
| $\delta(e)$ | Per-episode routing advantage |
| $\text{CR}_k$ | Top-$k$ concentration ratio |

---

*Draft version: 2026-05-16. For Paper Agent consumption.*
