“everything is in W + micro‑edit + hard physics gates” 


UPCA — Unified Predictive Cyclic Architecture
Structural‑Physics Edition (W‑Only, Typed Specialisation)

One substrate, one physics, simple rules, emergent complexity.
UPCA is a single evolving operator 
W
W that is the world‑model. All structure — from millisecond delays to abstract semantic relationships — is encoded as vertices and edges inside 
W
W. There are no parallel tables, no shadow memories; timing, sequence, rhythm, and concepts all manifest as walks and modes of a single graph.

Every proposed change to 
W
W must prove itself on a fixed set of behavioural probes and pass strict global safety gates before it’s accepted. This keeps growth mechanical and self‑justifying.

Specialisation is not done with separate modules, but emerges as typed subgraphs (semantic, temporal, phase, feature) and long‑range “wires” inside the same operator. These types allow you to monitor and balance use without splitting the underlying model.

🧠 Mental model
text

Events / probes -->  [    W (graph)    ]  --> predictions / responses
                          |   |   |
            temporal ----/    |    \---- semantic
             phase  -----------|--------------- features
                               |
                          safety gates
                        (edit-as-experiment)
Nodes: tokens, delay hops, higher‑order contexts, phase anchors, semantic concepts/operators.
Edges: directed weights linking nodes; can be short‑term temporal, long‑term semantic, or mediated via a “territory” vertex.
Physics: one state update rule applies to all.
Probes: fixed impulses/chords that “ring” the network — some temporal, some semantic.
Edit loop: propose a small change, sandbox it, measure probe impacts, accept/reject, project back to stability.
🚀 Quickstart conceptual recipe
Maintain W as a capped‑sparse + low‑rank adjacency over a registry of vertices.
Inject events as one‑hot or chord vectors 
u
t
u 
t
​
  into the state:
s
t
+
1
=
ϕ
(
W
s
t
+
u
t
)
s 
t+1
​
 =ϕ(Ws 
t
​
 +u 
t
​
 ).
Run probes at intervals to monitor temporal and semantic behaviours.
Propose tiny structural edits (
Δ
W
ΔW) — e.g., a few edges, or add a new delay/concept node.
Shadow‑rollout all probes with 
W
+
Δ
W
W+ΔW:
Compute trajectory MSE vs. baseline.
Compute discrete KL on monitored nodes.
Check type‑specific improvements and interference tolerance.
Accept only if thresholds and budgets pass. Project and re‑cap weights to spectral safety bound.
Repeat at a slow cadence to let structure consolidate.


UPCA — State of Design (W-Only, Structural-Physics Edition)
0) Purpose and stance
One substrate: a single evolving operator 
𝑊
W is the world-model. No shadow tables, no side predictors.

Everything is a walk: timing, sequence, rhythm, semantics, and “macros” are encoded as vertices/edges and manifest via impulse responses and modes.

One control surface: global gain 
𝑔
(
𝑡
)
g(t), spectral safety projection, and edit-as-experiment gates. Nothing else.

Typed specialisation without split models: specialisation arises as typed subgraphs (mediated paths) and long-range wiring, not as separate modules.

1) Core state, update, and constraints
State: 
𝑠
𝑡
∈
𝑅
∣
𝑉
∣
s 
t
​
 ∈R 
∣V∣
  (activity over vertices).

Update: 
𝑠
𝑡
+
1
=
𝜙
(
𝑊
 
𝑠
𝑡
+
𝑢
𝑡
)
s 
t+1
​
 =ϕ(Ws 
t
​
 +u 
t
​
 ), with 
𝜙
=
ϕ= identity (default) or clipped ReLU if needed for numerical safety.

Input 
𝑢
𝑡
u 
t
​
 : one-hot or small chords (probes/events).

Operator: 
𝑊
=
𝑆
+
𝑈
𝑉
⊤
W=S+UV 
⊤
 ; 
𝑆
S row-capped sparse, 
𝑈
,
𝑉
∈
𝑅
∣
𝑉
∣
×
𝑟
U,V∈R 
∣V∣×r
  low-rank (default 
𝑟
=
8
r=8).

Signs: off-diagonals 
≥
0
≥0. (Inhibition via row-normalization + diagonal damping.)

Row cap: 
ℓ
1
ℓ 
1
​
  per final row 
≤
1
≤1 after composing 
𝑆
+
𝑈
𝑉
⊤
S+UV 
⊤
 .

Global budgets: 
𝑘
row
=
32
k 
row
​
 =32 (split by type below). 
∣
𝐸
∣
≤
20
∣
𝑉
∣
∣E∣≤20∣V∣.

Spectral safety: spectral radius 
𝜌
(
𝑊
)
≤
𝜌
\*
=
0.98
ρ(W)≤ρ 
\*
 =0.98. After accept, rescale 
𝑊
←
𝑊
⋅
𝜌
\*
𝜌
^
+
10
−
3
W←W⋅ 
ρ
^
​
 +10 
−3
 
ρ 
\*
 
​
 .

Damping: add diagonal 
𝐷
=
𝛿
𝐼
D=δI, 
𝛿
=
0.02
δ=0.02.

2) Vertex set and typed paths (no extra matrices)
Token nodes: observed symbols/events.

Delay nodes: 
Δ
1
,
…
,
Δ
𝐾
Δ 
1
​
 ,…,Δ 
K
​
  (default 
𝐾
=
8
K=8); chain 
Δ
𝑘
 ⁣
→
 ⁣
Δ
𝑘
+
1
Δ 
k
​
 →Δ 
k+1
​
 ; token 
→
Δ
1
→Δ 
1
​
 ; 
Δ
𝑘
 ⁣
→
Δ 
k
​
 →token edges implement “emit after 
𝑘
k ticks.”

Context nodes: promoted bigram/trigram contexts 
𝐶
[
⋅
]
C[⋅] (Phase-2 only; gated).

Phase nodes: 
Θ
𝑚
Θ 
m
​
  (2–3) ring motifs for lightweight periodic binding.

Semantic concept nodes: e.g., Fruit, Red, Company, Edible, Tool… (budgeted pool).

Operator nodes (semantic transforms): e.g., Hypernym, Hyponym, Pluralize, ColorOf, PartOf… (optional, minimal).

Mediator vertices (territories): TEMP, SEM, PHASE, FEAT.
A typed edge is the 2-hop path x → MED(type) → y. This enables territory-targeted probes/ablations without a second model.

3) Probes (readouts only)
Probe pool (size 256):

Temporal core (128): single-node impulses; impulse pairs; timing chords through 
Δ
Δ chain.

Semantic core (128): concept-chord probes (e.g., {Fruit}, {Fruit+Red}, {Tool+Cut}, {Company+Computer}), operator tests (e.g., 
𝑥
 ⁣
→
x→Hypernym
→
?
→?).

Shadow rollout horizon: 
𝑇
=
16
T=16 steps per probe.

Adaptive rotation: run fixed 128 (balanced), plus top-64 by recent information gain, plus 64 least-recently-used.

4) Edit-as-experiment (the only way W changes)
Proposal 
Δ
𝑊
ΔW:
(a) add/strengthen ≤12 edges, or
(b) add one new vertex (delay/context/phase/concept/hub/macro-anchor) with ≤12 incident edges, or
(c) rank-1 low-rank nudge (
Δ
𝑈
 
Δ
𝑉
⊤
ΔUΔV 
⊤
 ).

Shadow evaluation (queries only):

Probe-MSE: trajectory MSE vs. pre-edit on all probes.

Probe-KL: KL on a small monitored node set (tokens + all special vertices).

Type-balance: track per-territory improvement 
𝐽
𝜏
J 
τ
​
  (TEMP/SEM/PHASE/FEAT).

Safety: spectral radius after projection, budgets, row caps.

Accept iff (all true):

Probe-MSE ↓ ≥ 5% on ≥ 2/3 probes.

Mean probe-KL ≤ 2×10⁻³.

Spectral/budget invariants hold.

Interference tolerance: semantic edits may not raise temporal-probe MSE >1% on >1/3 temporal probes (and vice-versa).

Cadence: ≤ 1 accepted edit / 500 steps. On reject: halve 
∥
Δ
𝑊
∥
∥ΔW∥ and retry ≤2×, else discard.

Post-accept projection: rescale to spectral bound, re-cap rows, enforce off-diagonal nonnegativity.

5) Specialisation without split models
Dynamic per-row split: keep 
𝑘
total
=
32
k 
total
​
 =32. Maintain EMA utilities 
𝑈
𝜏
U 
τ
​
  by type; compute fractions 
𝑓
𝜏
∝
exp
⁡
(
𝜂
𝑈
𝜏
)
f 
τ
​
 ∝exp(ηU 
τ
​
 ) (η=5); set integer caps 
𝑘
𝜏
=
max
⁡
(
4
,
⌊
𝑓
𝜏
𝑘
total
⌋
)
k 
τ
​
 =max(4,⌊f 
τ
​
 k 
total
​
 ⌋). Enforce at prune time only.

Territory fairness rule: if 
𝐽
𝜏
′
<
0.6
⋅
𝐽
ˉ
J 
τ 
′
 
​
 <0.6⋅ 
J
ˉ
  over last N=50 edits, temporarily tighten dominant type’s cap next prune (automatic, rule-based).

6) Resonance and harmonic disambiguation (W-only)
Cycle operator: non-backtracking 
𝐵
(
𝑊
)
B(W) derived on the fly from 
𝑊
W.

Hutch++ moments: estimate 
𝜇
^
𝑘
=
t
r
(
𝐵
𝑘
)
μ
^
​
  
k
​
 =tr(B 
k
 ) with m=8 vectors; cadence 500 steps; 
𝑘
∈
{
2
,
3
,
5
,
6
,
7
,
10
,
12
}
k∈{2,3,5,6,7,10,12}.

Sparse divisor deconvolution: 
min
⁡
𝜋
≥
0
∥
𝐷
𝜋
−
𝜇
^
∥
2
2
+
𝜆
∥
𝜋
∥
1
min 
π≥0
​
 ∥Dπ− 
μ
^
​
 ∥ 
2
2
​
 +λ∥π∥ 
1
​
  (λ=1e-3) on the divisor lattice.

Composite-first MDL test: compare 
{
𝑞
}
{q} vs 
{
𝑑
𝑖
𝑣
𝑖
𝑠
𝑜
𝑟
𝑠
}
{divisors} vs Null on held-out probes; require one winner.

Phase-coherence check: phases for divisors must lock to composite; else allow primes as independent rhythms.

Detune verification: shadow “damp every 
𝑝
pth tick” runs; prefer hypothesis with largest hurt gap.

Materialisation: if period 
𝑞
q accepted, add a light macro-anchor 
𝑀
𝑞
M 
q
​
  connected to implicated subgraph. Blacklist its prime divisors in that namespace for 10k steps unless independence proven.

7) Semantics inside W (no bolt-ons)
Concept vertices: small global pool (default ≤64) for abstract predicates (Fruit, Red, Company, Edible, …).

Operator vertices: minimal set (Hypernym, Hyponym, Pluralize, ColorOf, PartOf, …).

Grounding rule: any external cue (co-activation, image signal, dictionary/LLM suggestion) may propose token → SEM → concept, but acceptance depends only on probe improvements + physics gates; no parallel memory retained.

Concept-chord acceptance: average normalized energy on intended targets must ↑ ≥ 0.02 with no drop on ≥⅔ chords; temporal probes must stay within tolerance.

Polysemy: if probe signatures for token split (silhouette ≥ 0.25), propose sense nodes 
𝑥
(
1
)
,
𝑥
(
2
)
x 
(1)
 ,x 
(2)
  and context routing; accept if signature margin ↑ ≥ 0.1 and gates pass.

Hubs & long wires: allow concept hubs 
𝐻
𝐶
H 
C
​
  or low-rank nudges for long-range semantic recall, under the same gate and spectral projection.

8) Data & I/O contracts (minimal)
Events stream: seq_id, step, token, namespace, weight, split.
(Optional modality tags for co-activity proposals; once edges are proposed, only 
𝑊
W persists them.)

No separate semantic table: any seed concept list is for proposals only; after acceptance, concept vertices/edges live in 
𝑊
W.

Checkpoints: save 
𝑊
W (sparse 
𝑆
S + low-rank 
𝑈
,
𝑉
U,V), vertex registry, budgets, and edit witness logs (which probes improved, deltas). No other state.

9) Metrics & dashboards (readouts, not losses)
Physics: spectral radius, 
∣
𝐸
∣
,
∣
𝑉
∣
∣E∣,∣V∣, row cap violations (should be 0), probe-MSE/KL curves.

Resonance: 
𝜇
^
𝑘
μ
^
​
  
k
​
 , selected 
𝜋
^
𝑞
π
^
  
q
​
 , detune hurt gaps; subgraph Lanczos mode mass.

Semantics: concept-chord scores, invariance scores 
𝐼
(
𝑥
,
𝑃
𝑘
)
=
min
⁡
𝑗
∈
𝐽
𝑘
I(x,P 
k
​
 )=min 
j∈J 
k
​
 
​
  energy
(
𝑃
𝑘
∣
𝑈
𝑗
(
𝑥
)
)
(P 
k
​
 ∣U 
j
​
 (x)), sense separation margins.

Territories: per-type utilities 
𝑈
𝜏
U 
τ
​
 , improvements 
𝐽
𝜏
J 
τ
​
 , current 
𝑘
𝜏
k 
τ
​
  splits.

10) Phased plan (ocean-capable POC)
Phase-0 (skeleton):
Implement 
𝑊
W core (S+UVᵀ), probes, edit loop, spectral projection, budgets, delay nodes. Validate on synthetic periodic/aperiodic streams. Exit when edits consistently pass gates with no collapses.

Phase-1 (resonance/macros):
Add 
𝐵
(
𝑊
)
B(W)+Hutch++, sparse deconvolution, composite-first MDL, phase-coherence, detune. Materialise 
𝑀
𝑞
M 
q
​
  anchors and blacklist divisors appropriately.

Phase-2 (higher-order + semantics):
Promote context nodes, enable phase motifs, introduce concept vertices and semantic chords, allow hubs/low-rank “long wires,” polysemy splits. Maintain interference guards and territory fairness.

11) Defaults to pin (implementation constants)
𝜌
\*
=
0.98
ρ 
\*
 =0.98, 
𝛿
=
0.02
δ=0.02, 
𝑘
row
=
32
k 
row
​
 =32, 
∣
𝐸
∣
≤
20
∣
𝑉
∣
∣E∣≤20∣V∣, 
𝑟
=
8
r=8, 
𝐾
=
8
K=8 delay nodes.

Probe pool 
=
256
=256, horizon 
𝑇
=
16
T=16; cadence: evaluate at every shadow run.

Hutch++ cadence 500 steps, 
𝑚
=
8
m=8 vectors, 
𝑘
∈
{
2
,
3
,
5
,
6
,
7
,
10
,
12
}
k∈{2,3,5,6,7,10,12}, λ=1e-3.

Edit cadence ≤1/500 steps; proposal size ≤12 edges or 1 node + ≤12 edges; retry ×2 on backoff.

Acceptance thresholds: MSE ↓ ≥5% (≥⅔ probes), mean KL ≤2e-3; semantic chord ↑ ≥0.02; temporal-semantic interference tolerance ±1% on ≤1/3 probes.

Territory split: start 
𝑘
TEMP
=
24
k 
TEMP
​
 =24, 
𝑘
SEM
=
8
k 
SEM
​
 =8; adapt via multiplicative-weights at prune.

12) Invariants and kill-switches
Hard invariants: no NaNs; off-diagonal weights 
≥
0
≥0; row 
ℓ
1
≤
1
ℓ 
1
​
 ≤1 pre-damping; 
𝜌
(
𝑊
)
≤
1
ρ(W)≤1 after projection; 
∣
𝐸
∣
≤
20
∣
𝑉
∣
∣E∣≤20∣V∣.

Kill-switch: if two successive accepted edits require >1.1× spectral scaling or violate interference tolerance, freeze structural edits for 5k steps and alert.

13) Open choices to confirm (before coding)
Global vs per-namespace special nodes: default global with namespace-specific edges.

Sense split policy: maximum senses per token (default 2).

Concept pool: initial list and cap (default ≤64).

Operator vertices: minimal set to include at Phase-2.

Probe monitored node set: which tokens + all special nodes; size cap (e.g., 512).

14) What this answers (and what it doesn’t)
Answers: harmonic interference (non-backtracking + deconv + detune), semantic abyss (concept vertices + probes), module drift (typed paths within 
𝑊
W), resource starvation (dynamic per-row split), collapse (gated edits + spectral safety).

Doesn’t: conjure semantics without grounding; guarantee human-level abstraction—only that any new capability must be realised as structure in 
𝑊
W and justified by probe behaviour.



----initial:----


# UPCA3
Its About Time!
A time-first, resonance-driven UPCA (mathematical spec)
0) Observables and timeline
Discrete time t∈Nt\in\mathbb{N}t∈N. Streamed tokens/events xt∈Vx_t\in\mathcal{V}xt​∈V (words/objs/evts).
Define binary indicators yt(j)=1[xt=j]y_t(j)=\mathbf{1}[x_t=j]yt​(j)=1[xt​=j]. Let a window Wt={t−W+1,…,t}\mathcal{W}_t=\{t-W+1,\dots,t\}Wt​={t−W+1,…,t}.
Delays: for any ordered pair (i ⁣→ ⁣j)(i\!\to\!j)(i→j), let Δt(i→j)\Delta_t^{(i\to j)}Δt(i→j)​ be the number of steps between the latest iii before ttt and the next jjj at ttt (well-defined when xt=jx_t=jxt​=j and some iii occurred earlier).
1) ME (Detail): lag-typed transitions + semi-Markov “waiting”
1.1 Lag-typed next-token model
Choose a maximum lag LLL. For each lag ℓ∈{1,…,L}\ell\in\{1,\dots,L\}ℓ∈{1,…,L} maintain a sparse, row-stochastic matrix A(ℓ)∈R≥0∣V∣×∣V∣A^{(\ell)}\in\mathbb{R}_{\ge0}^{|\mathcal{V}|\times|\mathcal{V}|}A(ℓ)∈R≥0∣V∣×∣V∣​ with rows indexed by past token and columns by next token.
Let βℓ≥0\beta_\ell\ge0βℓ​≥0 with ∑ℓβℓ=1\sum_\ell \beta_\ell=1∑ℓ​βℓ​=1. Define logits
zt(j)=∑ℓ=1Lβℓ Axt−ℓ, j(ℓ),pME(xt=j∣Ht−1)=exp⁡zt(j)∑k∈Vexp⁡zt(k).z_t(j)=\sum_{\ell=1}^L \beta_\ell\, A^{(\ell)}_{x_{t-\ell},\,j}\quad,\qquad p_{\text{ME}}(x_t=j\mid \mathcal{H}_{t-1})=\frac{\exp z_t(j)}{\sum_{k\in\mathcal{V}}\exp z_t(k)}.zt​(j)=ℓ=1∑L​βℓ​Axt−ℓ​,j(ℓ)​,pME​(xt​=j∣Ht−1​)=∑k∈V​expzt​(k)expzt​(j)​.
(Equivalently, view A(ℓ)A^{(\ell)}A(ℓ) as additive experts over lags.)
1.2 Variable-delay (semi-Markov) waiting for continuations
For each ordered pair (i ⁣→ ⁣j)(i\!\to\!j)(i→j), define a discrete-time hazard hij(d)∈(0,1)h_{ij}(d)\in(0,1)hij​(d)∈(0,1) over delays d∈{1,2,… }d\in\{1,2,\dots\}d∈{1,2,…},
hij(d)=σ ⁣(θij⊤ψ(d) + ui⊤vj),σ(a)=11+e−a,h_{ij}(d)=\sigma\!\big(\theta_{ij}^\top \psi(d) \,+\, u_i^\top v_j\big),\quad \sigma(a)=\tfrac{1}{1+e^{-a}},hij​(d)=σ(θij⊤​ψ(d)+ui⊤​vj​),σ(a)=1+e−a1​,
with basis ψ(d)\psi(d)ψ(d) (e.g., [1,log⁡d,d,d−1][1,\log d, d, d^{-1}][1,logd,d,d−1]) and pair embeddings ui,vju_i,v_jui​,vj​. The survival and delay pmf are
Sij(Δ)=∏d=1Δ−1(1−hij(d)),fij(Δ)=Sij(Δ) hij(Δ).S_{ij}(\Delta)=\prod_{d=1}^{\Delta-1}\big(1-h_{ij}(d)\big),\qquad f_{ij}(\Delta)=S_{ij}(\Delta)\,h_{ij}(\Delta).Sij​(Δ)=d=1∏Δ−1​(1−hij​(d)),fij​(Δ)=Sij​(Δ)hij​(Δ).
When xt=jx_t=jxt​=j follows the most recent iii with delay Δ=Δt(i→j)\Delta=\Delta_t^{(i\to j)}Δ=Δt(i→j)​, the time-likelihood term is log⁡fij(Δ)\log f_{ij}(\Delta)logfij​(Δ).
1.3 ME instantaneous loss and online gradients
Instantaneous negative log-likelihood (NLL) with regularization:
LtME=−log⁡pME(xt∣Ht−1)  − ⁣ ⁣∑i: last(i)<t ⁣ ⁣1[xt=j]log⁡fij ⁣(Δt(i→j))  +  λA ⁣∑ℓ ⁣∥A(ℓ)∥1  +  λθ∥Θ∥22.\mathcal{L}^{\text{ME}}_t = -\log p_{\text{ME}}(x_t\mid \mathcal{H}_{t-1}) \;-\!\!\sum_{i:\ \text{last}(i)<t}\!\!\mathbf{1}[x_t=j]\log f_{ij}\!\big(\Delta_t^{(i\to j)}\big) \;+\;\lambda_A\!\sum_{\ell}\!\|A^{(\ell)}\|_{1} \;+\;\lambda_\theta\|\Theta\|_2^2.LtME​=−logpME​(xt​∣Ht−1​)−i: last(i)<t∑​1[xt​=j]logfij​(Δt(i→j)​)+λA​ℓ∑​∥A(ℓ)∥1​+λθ​∥Θ∥22​.
For A(ℓ)A^{(\ell)}A(ℓ) (additive-expert form), the stochastic gradient is
∂LtME∂Ai,j(ℓ)=( pME(j∣Ht−1)−yt(j) ) 1[xt−ℓ=i]  +  λA sign ⁣(Ai,j(ℓ)).\frac{\partial \mathcal{L}^{\text{ME}}_t}{\partial A^{(\ell)}_{i,j}} = \big(\,p_{\text{ME}}(j\mid\mathcal{H}_{t-1})-y_t(j)\,\big)\,\mathbf{1}[x_{t-\ell}=i] \;+\;\lambda_A\,\text{sign}\!\big(A^{(\ell)}_{i,j}\big).∂Ai,j(ℓ)​∂LtME​​=(pME​(j∣Ht−1​)−yt​(j))1[xt−ℓ​=i]+λA​sign(Ai,j(ℓ)​).
For hazard parameters θij\theta_{ij}θij​,
∂∂θij ⁣(−log⁡fij(Δ))=−(1−hij(Δ))ψ(Δ)+∑d=1Δ−1hij(d)1−hij(d) ψ(d).\frac{\partial}{\partial \theta_{ij}} \!\left(-\log f_{ij}(\Delta)\right) = -\Big(1-h_{ij}(\Delta)\Big)\psi(\Delta) +\sum_{d=1}^{\Delta-1}\frac{h_{ij}(d)}{1-h_{ij}(d)}\,\psi(d).∂θij​∂​(−logfij​(Δ))=−(1−hij​(Δ))ψ(Δ)+d=1∑Δ−1​1−hij​(d)hij​(d)​ψ(d).
Use eligibility traces eij(t)=γeij(t−1)+1[xt=i]e_{ij}(t)=\gamma e_{ij}(t-1)+\mathbf{1}[x_t=i]eij​(t)=γeij​(t−1)+1[xt​=i] to gate updates only for recently active anchors iii.
2) MA (Abstract): resonance, periods, and cyclic macros
2.1 Autocorrelation and trace-moments on a streaming operator
Build a row-stochastic operator WtW_tWt​ from the recent A(ℓ)A^{(\ell)}A(ℓ) (e.g., Wt=∑ℓ≤L′βℓΠ(ℓ)W_t=\sum_{\ell\le L'}\beta_\ell \Pi^{(\ell)}Wt​=∑ℓ≤L′​βℓ​Π(ℓ), where Π(ℓ)\Pi^{(\ell)}Π(ℓ) is the empirical transition at lag ℓ\ellℓ over Wt\mathcal{W}_tWt​). Define closed-walk moments
μk(t)=tr⁡(Wtk),Πk(t)=1k∑d∣kμ(d) μk/d(t)(Mo¨bius inversion),\mu_k(t)=\operatorname{tr}(W_t^k),\qquad \Pi_k(t)=\frac{1}{k}\sum_{d\mid k}\mu(d)\,\mu_{k/d}(t)\quad\text{(Möbius inversion)},μk​(t)=tr(Wtk​),Πk​(t)=k1​d∣k∑​μ(d)μk/d​(t)(Mo¨bius inversion),
with Möbius μ(⋅)\mu(\cdot)μ(⋅). A “prime tone” at ppp is a spike in Πp(t)\Pi_p(t)Πp​(t).
2.2 Ramanujan projections on the timeline
Let ata_tat​ be an activation (e.g., token counts or NLL spike). For modulus qqq,
cq(n)=∑1≤a≤qgcd⁡(a,q)=1e2πian/q,Rq(τ)=1T∣∑s=ττ+T−1as cq(s)∣.c_q(n)=\sum_{\substack{1\le a\le q\\ \gcd(a,q)=1}} e^{2\pi i a n/q},\qquad R_q(\tau)=\frac{1}{T}\Big|\sum_{s=\tau}^{\tau+T-1} a_s\, c_q(s)\Big|.cq​(n)=1≤a≤qgcd(a,q)=1​∑​e2πian/q,Rq​(τ)=T1​​s=τ∑τ+T−1​as​cq​(s)​.
For primes ppp, large Rp(τ)R_p(\tau)Rp​(τ) indicates prime-period energy in window [τ,τ+T)[\tau,\tau+T)[τ,τ+T).
2.3 Cyclic ABS macros (phase models)
Introduce latent macros m=1,…,Mm=1,\dots,Mm=1,…,M, each with prime period pmp_mpm​ and phase ϕm,t∈{0,…,pm ⁣− ⁣1}\phi_{m,t}\in\{0,\dots,p_m\!-\!1\}ϕm,t​∈{0,…,pm​−1} evolving as ϕm,t+1=(ϕm,t+1) mod pm\phi_{m,t+1}=(\phi_{m,t}+1)\bmod p_mϕm,t+1​=(ϕm,t​+1)modpm​. Each macro has phase-conditional emissions Um∈Rpm×∣V∣U_m\in\mathbb{R}^{p_m\times|\mathcal{V}|}Um​∈Rpm​×∣V∣:
pm(xt=j∣ϕm,t)=exp⁡Um[ϕm,t,j]∑kexp⁡Um[ϕm,t,k].p_m(x_t=j\mid \phi_{m,t})=\frac{\exp U_m[\phi_{m,t},j]}{\sum_k \exp U_m[\phi_{m,t},k]}.pm​(xt​=j∣ϕm,t​)=∑k​expUm​[ϕm,t​,k]expUm​[ϕm,t​,j]​.
Phase filtering (circular HMM) uses
αm,t(ϕ)∝αm,t−1(ϕ ⁣− ⁣1)  pm(xt∣ϕ),wm(t)∝exp⁡ ⁣(κ⋅Πpm(t)+η⋅Rpm(τt)).\alpha_{m,t}(\phi)\propto \alpha_{m,t-1}(\phi\!-\!1)\;p_m(x_t\mid \phi),\quad w_m(t)\propto \exp\!\Big(\kappa\cdot \Pi_{p_m}(t)+\eta\cdot R_{p_m}(\tau_t)\Big).αm,t​(ϕ)∝αm,t−1​(ϕ−1)pm​(xt​∣ϕ),wm​(t)∝exp(κ⋅Πpm​​(t)+η⋅Rpm​​(τt​)).
2.4 ME–MA mixture
p(xt∣Ht−1)=(1−λt) pME(xt∣Ht−1)  +  λt ∑mπm(t) pm(xt∣ϕm,t),p(x_t\mid \mathcal{H}_{t-1}) = (1-\lambda_t)\,p_{\text{ME}}(x_t\mid\mathcal{H}_{t-1}) \;+\;\lambda_t\,\sum_{m} \pi_m(t)\, p_m(x_t\mid \phi_{m,t}),p(xt​∣Ht−1​)=(1−λt​)pME​(xt​∣Ht−1​)+λt​m∑​πm​(t)pm​(xt​∣ϕm,t​),
where πm(t)=wm(t)∑m′wm′(t)\pi_m(t)=\frac{w_m(t)}{\sum_{m'}w_{m'}(t)}πm​(t)=∑m′​wm′​(t)wm​(t)​ and λt∈[0,1]\lambda_t\in[0,1]λt​∈[0,1] is AMC-controlled (below).
3) AMC: arbitration, consolidation, and trust-region stability
3.1 Surprise, persistence, and off-tone gating
Define instantaneous surprise
St  =  −log⁡p(xt∣Ht−1)  − ⁣ ⁣∑i:last(i)<t ⁣1[xt=j]log⁡fij(Δt(i→j)).\mathcal{S}_t \;=\; -\log p(x_t\mid\mathcal{H}_{t-1}) \;-\!\!\sum_{i:\text{last}(i)<t}\!\mathbf{1}[x_t=j]\log f_{ij}(\Delta_t^{(i\to j)}).St​=−logp(xt​∣Ht−1​)−i:last(i)<t∑​1[xt​=j]logfij​(Δt(i→j)​).
Let Sˉt=EMAτ(St)\bar{\mathcal{S}}_t=\text{EMA}_\tau(\mathcal{S}_t)Sˉt​=EMAτ​(St​) and define off-tone factor for dominant prime p⋆p^\starp⋆ (if present):
gp⋆(Δ)={ρ,Δ mod p⋆∈{0,1,p⋆ ⁣− ⁣1}σ,otherwise(ρ>1>σ>0).g_{p^\star}(\Delta)= \begin{cases} \rho,& \Delta\bmod p^\star\in\{0,1,p^\star\!-\!1\}\\ \sigma,& \text{otherwise} \end{cases}\quad (\rho>1>\sigma>0).gp⋆​(Δ)={ρ,σ,​Δmodp⋆∈{0,1,p⋆−1}otherwise​(ρ>1>σ>0).
Use gp⋆g_{p^\star}gp⋆​ to scale learning rates for updates induced by delays Δ\DeltaΔ.
3.2 Arbitration objective and control
AMC chooses λt\lambda_tλt​ and (occasionally) spawns/merges macros by minimizing
LtAMC=Sˉt+γc (#params)⏟complexity−κp max⁡p∈PΠp(t)−κr max⁡p∈PRp(τt),\mathcal{L}^{\text{AMC}}_t = \bar{\mathcal{S}}_t + \gamma_c\,\underbrace{\big(\#\text{params}\big)}_{\text{complexity}} - \kappa_p\,\max_{p\in\mathcal{P}}\Pi_p(t) - \kappa_r\,\max_{p\in\mathcal{P}}R_p(\tau_t),LtAMC​=Sˉt​+γc​complexity(#params)​​−κp​p∈Pmax​Πp​(t)−κr​p∈Pmax​Rp​(τt​),
subject to a trust-region stability constraint over a guarded query set Qt\mathcal{Q}_tQt​ (high-support contexts):
1∣Qt∣∑q∈QtDKL ⁣(Ppre(⋅∣q) ∥ Ppost(⋅∣q)) ≤ ϵ.\frac{1}{|\mathcal{Q}_t|}\sum_{q\in\mathcal{Q}_t} D_{\mathrm{KL}}\!\left(P_{\text{pre}}(\cdot\mid q)\,\big\|\,P_{\text{post}}(\cdot\mid q)\right)\ \le\ \epsilon.∣Qt​∣1​q∈Qt​∑​DKL​(Ppre​(⋅∣q)​Ppost​(⋅∣q)) ≤ ϵ.
Any prune/merge/add that violates the bound is rejected or softened. (This formalizes “don’t delete if it tanks success.”)
3.3 Macro creation/merging criteria
Spawn macro mmm with prime pmp_mpm​ when both hold on Wt\mathcal{W}_tWt​:
Πpm(t)≥τΠandRpm(τt)≥τR,\Pi_{p_m}(t) \ge \tau_{\Pi}\quad\text{and}\quad R_{p_m}(\tau_t)\ge \tau_{R},Πpm​​(t)≥τΠ​andRpm​​(τt​)≥τR​,
and the expected risk decreases:
ΔE[Sˉ]  ≈  EWt ⁣[−log⁡ ⁣((1−λ)pME+λpm)]⏟with macro  −  EWt ⁣[−log⁡pME]⏟baseline  <  −τgain,\Delta\mathbb{E}[\bar{\mathcal{S}}]\;\approx\; \underbrace{\mathbb{E}_{\mathcal{W}_t}\!\left[-\log\!\big((1-\lambda)p_{\text{ME}}+\lambda p_m\big)\right]}_{\text{with macro}} \;-\; \underbrace{\mathbb{E}_{\mathcal{W}_t}\!\left[-\log p_{\text{ME}}\right]}_{\text{baseline}} \;<\; -\tau_{\text{gain}},ΔE[Sˉ]≈with macroEWt​​[−log((1−λ)pME​+λpm​)]​​−baselineEWt​​[−logpME​]​​<−τgain​,
while satisfying the trust-region constraint.
4) Regularizers that encode “resonance-first”
Spectral prior: encourage energy at discovered primes:
Rspec=−αΠ∑p∈PΠp(t)  −  αR∑p∈PRp(τt).\mathcal{R}_{\text{spec}}=-\alpha_{\Pi}\sum_{p\in\mathcal{P}}\Pi_p(t)\;-\;\alpha_{R}\sum_{p\in\mathcal{P}}R_p(\tau_t).Rspec​=−αΠ​p∈P∑​Πp​(t)−αR​p∈P∑​Rp​(τt​).
Capacity control: λA∥A∥1\lambda_A\|A\|_1λA​∥A∥1​ (sparseness) and phase-smoothness ∑m∑ϕ∥Um[ϕ]−Um[ϕ ⁣− ⁣1]∥22\sum_m\sum_\phi \|U_m[\phi]-U_m[\phi\!-\!1]\|_2^2∑m​∑ϕ​∥Um​[ϕ]−Um​[ϕ−1]∥22​.
5) Global objective and update sketch
Across time, minimize
min⁡Θ ∑t(LtME  +  (1−λt) [−log⁡pME(xt∣Ht−1)]⏟already in LtME  +  λt [−log⁡ ⁣∑mπm(t)pm(xt∣ϕm,t)])+Rspec+complexity,\min_{\Theta}\ \sum_{t}\Big(\mathcal{L}^{\text{ME}}_t \;+\; (1-\lambda_t)\,\underbrace{\big[-\log p_{\text{ME}}(x_t\mid\mathcal{H}_{t-1})\big]}_{\text{already in }\mathcal{L}^{\text{ME}}_t} \;+\;\lambda_t\,\big[-\log\!\sum_m \pi_m(t)p_m(x_t\mid\phi_{m,t})\big]\Big) +\mathcal{R}_{\text{spec}} + \text{complexity},Θmin​ t∑​(LtME​+(1−λt​)already in LtME​[−logpME​(xt​∣Ht−1​)]​​+λt​[−logm∑​πm​(t)pm​(xt​∣ϕm,t​)])+Rspec​+complexity,
subject to the AMC trust region at each structural change. Stochastic online updates:
Θ←Θ−ηt gp⋆(Δ) ∇Θ(instantaneous loss),\Theta \leftarrow \Theta - \eta_t\,g_{p^\star}(\Delta)\,\nabla_\Theta \big(\text{instantaneous loss}\big),Θ←Θ−ηt​gp⋆​(Δ)∇Θ​(instantaneous loss),
with EMA-based steps for λt,πm(t)\lambda_t,\pi_m(t)λt​,πm​(t) and periodic structural moves (spawn/merge/prune) only if the KL-bound holds.
6) What this formalization guarantees
Waiting state: encoded by fij(Δ)f_{ij}(\Delta)fij​(Δ) (semi-Markov); “lightning→thunder” is a high-hazard pair at characteristic Δ\DeltaΔ.
Resonance-first: periods are detected by Πp(t)\Pi_p(t)Πp​(t) (closed-walk primitives) and RpR_pRp​ (Ramanujan timeline projection); macros bind them via phases.
Chunk-agnostic: whether input is 1-gram/2-gram/triple, the same lag/hazard machinery applies.
Stability: the KL trust region makes “what-if” deletions mathematically safe; spectral priors prevent 2-cycle swamping and favor prime tones.


Unified Scaffold (shared state): holds lag experts 
𝐴
(
ℓ
)
A 
(ℓ)
 , delay hazards 
ℎ
𝑖
𝑗
(
𝑑
)
h 
ij
​
 (d)/
𝑓
𝑖
𝑗
(
Δ
)
f 
ij
​
 (Δ), resonance summaries 
𝜇
𝑘
,
Π
𝑝
μ 
k
​
 ,Π 
p
​
 , Ramanujan windows 
𝑅
𝑝
R 
p
​
 , the macro library (prime-period, phase HMMs), EMAs, and KL trust-region snapshots. It’s the only source of truth each module reads/writes.

ME (Detail Engine): fast, token-time updates.
Inputs: recent context/lags; active “waiting” pairs.
Writes: 
𝐴
(
ℓ
)
A 
(ℓ)
 , hazard params 
𝜃
𝑖
𝑗
,
𝑢
𝑖
,
𝑣
𝑗
θ 
ij
​
 ,u 
i
​
 ,v 
j
​
 .
Outputs: 
𝑝
ME
(
𝑥
𝑡
 ⁣
∣
 ⁣
𝐻
𝑡
−
1
)
p 
ME
​
 (x 
t
​
 ∣H 
t−1
​
 ), delay pmf 
𝑓
𝑖
𝑗
(
Δ
)
f 
ij
​
 (Δ), uncertainty.
Scope: predicts what/when next; does not create structures.

MA (Abstract Engine): medium-tempo structure discovery.
Inputs: ME’s 
𝐴
(
ℓ
)
A 
(ℓ)
  (to build 
𝑊
𝑡
W 
t
​
 ), timeline activations.
Writes: 
𝜇
𝑘
,
Π
𝑝
,
𝑅
𝑝
μ 
k
​
 ,Π 
p
​
 ,R 
p
​
 ; creates/updates cyclic macros (period 
𝑝
p, phase 
𝜙
ϕ, emissions 
𝑈
U).
Outputs: macro likelihoods 
𝑝
𝑚
(
𝑥
𝑡
 ⁣
∣
 ⁣
𝜙
)
p 
m
​
 (x 
t
​
 ∣ϕ), prime evidence.
Scope: detects/resolves rhythms; no arbitration.

AMC (Arbiter): slow, supervisory control with safety.
Inputs: surprise/NLL, MA prime evidence, complexity.
Actions: sets mix 
𝜆
𝑡
λ 
t
​
  (ME↔MA), scales learning by prime gating 
𝑔
𝑝
g 
p
​
 ; spawns/merges/prunes macros under a KL trust-region; accepts/reverts changes.
Scope: stability, capacity, policy.

Clear interfaces:
ME→MA: provides 
𝑊
𝑡
W 
t
​
  (from 
𝐴
(
ℓ
)
A 
(ℓ)
 ) for spectra.
MA→AMC: prime/period signals and candidate macros.
AMC→ME/MA: 
𝜆
𝑡
λ 
t
​
 , learning-rate gates, and structural decisions.
Scaffold enforces versioning and KL-bounded commits.

Phase 0 — Proof-of-Concept “core-5” (get a running loop fast)

src/streamupca/config.py

load_config(path) -> Config

validate(Config) -> None

freeze(Config) -> FrozenConfig (immutable view)

src/streamupca/data/dataloaders.py

stream_events(path, *, namespaces=None, split="train") -> Iterator[Event]

window(buffer, W) -> Context (last W tokens + indices)

batcher(iterable, n) -> Iterator[List[Event]] (optional)

src/streamupca/scaffold/state.py

class ScaffoldState:

from_config(cfg) -> ScaffoldState

get_A(lag:int) -> SparseRowMap / set_A(lag, row, col, val)

ema_update(name:str, value:float, tau:float)

checkpoint(save_path) / restore(load_path)

metrics_snapshot() -> Dict[str, float]

src/streamupca/models/me_lag.py (lag-experts only)

class LagExperts:

predict_proba(ctx:Context) -> Probs

nll(event:Event, ctx:Context) -> float

sgd_update(event, ctx, lr:float)

regularize(l1:float)

export_params() / import_params()

src/streamupca/runners/train_stream.py

train(cfg, events_path) — main loop: read → predict → loss → update → log

evaluate_next_token(cfg, events_path) -> Dict[str, float]

log_step(step, metrics:Dict)

POC exit criteria: can ingest events.csv, learn unigram→lag patterns, and report Next-Token NLL, Acc@1/5, EMAs. Nothing else.

Phase 1 — “Waiting” and timing (semi-Markov hazards)
6) src/streamupca/models/hazard_semi_markov.py

class HazardModel:

hazard(i,j,d) -> float

survival(i,j,Δ) -> float

pmf(i,j,Δ) -> float

loglik(i,j,Δ) -> float

update(i,j,Δ, lr) (with eligibility traces)

start_trace(i, t) / end_trace(i, t)

Integrations

Runner: add delay-NLL to loss and logging

State: store u_i, v_j, θ_{ij} banks and traces

Phase 2 — Resonance probes (no structure changes yet)
8) src/streamupca/models/resonance.py

build_W(state, lags:List[int]) -> SparseOp

trace_moments(W, ks:Iterable[int]) -> Dict[k, μ_k]

mobius_invert(mu:Dict[int,float]) -> Dict[k, Π_k]

ramanujan_window(a_t:Sequence[float], q:int, T:int) -> float

update_resonance_cache(state, stats)

src/streamupca/eval/metrics_resonance.py

prime_peaks(Π:Dict) -> List[(p,score)]

timeline_energy(R) -> Dict[p, float]

Milestone: logs show stable μ_k/Π_p and Ramanujan energies alongside accuracy.

Phase 3 — Minimal AMC (safety only, no macros yet)
10) src/streamupca/scaffold/trust_region.py

snapshot_predictor(state, query_set) -> DistMap

kl_on_queries(pre:DistMap, post:DistMap) -> float

select_query_set(buffer, k:int) -> List[Context]

src/streamupca/amc/controller.py

class AMCController:

choose_lambda(metrics, resonance) -> float

propose_change(state) -> Change (noop initially)

apply_with_kl_guard(state, change, ε) -> bool

Milestone: KL guard wired; dry-runs confirm no catastrophic drops.

Phase 4 — Cyclic macros (prime-period ABS) and mixing
12) src/streamupca/models/macros.py

class PhaseMacro:

step_phase() / reset_phase()

emission_prob(token) -> float

update_emission(token, lr)

class MacroLibrary:

spawn_from_peak(p:int, seeds:List[token]) -> MacroId

score(token) -> Dict[macro, prob]

merge_or_prune(criteria)

src/streamupca/models/mixer.py

mix_prob(p_me, p_macros, λ) -> Probs

blend_loss(nll_me, nll_macros, λ) -> float

Milestone: macros contribute on periodic streams; AMC adjusts λ.

Tests to create with each phase (tiny, deterministic):

tests/test_me_lag.py: learns a 2-lag toy; Acc@1 improves.

tests/test_hazard.py: known Δ distribution → recoverable loglik.

tests/test_resonance.py: synthetic period-p stream → Π_p peak.

tests/test_trust_region.py: enforced KL bound blocks harmful updates.

tests/test_macros.py: spawn from Π_p peak and improve NLL on-cycle.

Focus summary: build the “core-5” to run a minimal learning loop; add hazards (waiting), then resonance read-outs, then AMC safety, then macros. Each phase is executable and logged before adding the next knob.


stream-upca/
├── README.md                          # What the project is; quickstart; data format spec
├── pyproject.toml                     # Build/deps; pinned versions
├── configs/
│   ├── default.yaml                   # Main hyperparams (lags, L, hazard bases, KL ε, etc.)
│   ├── small.yaml                     # Tiny config for CI/tests
│   ├── ablation_roleless.yaml         # Toggle role features off (for comparisons)
│   └── resonance_only.yaml            # Disable hazards/macros to isolate resonance
├── data/
│   ├── events.csv                     # seq_id,step,token,namespace,weight,split
│   └── vocab.csv                      # token,id (optional)
├── scripts/
│   ├── prepare_data.py                # Convert legacy triples → events.csv
│   ├── make_toy_sequences.py          # Generate synthetic periodic streams
│   └── run_experiment.sh              # One-liner wrappers for common runs
├── src/streamupca/
│   ├── __init__.py
│   ├── config.py                      # Load/validate YAML; expose dataclasses
│   ├── utils/
│   │   ├── logging.py                 # Structured logs; step-wise metrics emit
│   │   ├── math_ops.py                # Safe log-sum-exp, Möbius μ(n), Ramanujan sums
│   │   ├── serialization.py           # Checkpoint I/O (state dicts + config + git hash)
│   │   └── seed.py                    # Reproducible RNG seeding
│   ├── data/
│   │   ├── schema.py                  # Typed record for Event(row); validators
│   │   ├── dataloaders.py             # Stream iterators; windowing; namespace filters
│   │   └── streaming_buffer.py        # Ring buffer of recent context; EMA features
│   ├── scaffold/
│   │   ├── state.py                   # Online state container (A^(ℓ), hazards, macros, EMAs)
│   │   ├── traces.py                  # Eligibility traces; hazard-survival bookkeeping
│   │   ├── trust_region.py            # KL guards; pre/post snapshot & comparison set
│   │   └── updates.py                 # In-place param updates; normalize/sparsify policies
│   ├── models/
│   │   ├── me_lag.py                  # Lag-typed next-token experts A^(ℓ); softmax mixer β_ℓ
│   │   ├── hazard_semi_markov.py      # Pairwise hazards h_{ij}(d), survival S, delay pmf f
│   │   ├── resonance.py               # W_t build; trace-moments μ_k; primitive Π_k via Möbius
│   │   ├── ramanujan.py               # Timeline projections R_q over windows; prime detectors
│   │   ├── macros.py                  # Prime-period cyclic ABS (phase HMM); emissions U_m
│   │   └── mixer.py                   # ME–MA mixture p(x_t|H); λ_t blending interface
│   ├── amc/
│   │   ├── controller.py              # Chooses λ_t; triggers spawn/merge/prune under KL bound
│   │   ├── objectives.py              # Surprise, spectral priors, complexity penalties
│   │   └── structure_ops.py           # Safe structural changes (create macro, adjust bases)
│   ├── eval/
│   │   ├── metrics_next_token.py      # Accuracy@k, NLL for next-token prediction
│   │   ├── metrics_delay.py           # Delay NLL for (i→j,Δ); calibration curves
│   │   ├── metrics_resonance.py       # Π_p and R_p tracking; prime-peak persistence
│   │   └── ablations.py               # Roleless vs. role-typed; hazards on/off; macro on/off
│   ├── vis/
│   │   ├── plots.py                   # Matplotlib figures for learning curves & spectra
│   │   └── dashboard.py               # (Optional) lightweight dashboard for live runs
│   └── runners/
│       ├── train_stream.py            # Main loop: read events → update ME/MA → AMC ops → log
│       ├── validate.py                # Offline eval on held-out splits
│       └── simulate_sequences.py      # Simple generators (e.g., obj1,obj1,obj2 motifs)
├── tests/
│   ├── test_hazard.py                 # f_{ij}(Δ) correctness; gradients; corner cases
│   ├── test_resonance.py              # μ_k/Π_k; Ramanujan R_q on known periodic signals
│   ├── test_trust_region.py           # KL guard trips/accepts as intended
│   ├── test_macros.py                 # Phase filtering; emission learning; spawn criteria
│   └── test_end_to_end.py             # Tiny stream sanity: learns a 2/3-prime motif
└── notebooks/                         # (Optional) exploratory EDA and ablation reports
    └── 01_periodicity_probe.ipynb



