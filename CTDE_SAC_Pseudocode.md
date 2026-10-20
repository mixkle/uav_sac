# Algorithm 1. Parameter-Shared Multi-Agent SAC for UAV Deployment

**Input:** episode count E, horizon T, batch size K, replay threshold L_start ≥ K, discount β, soft-update coefficient η, and joint target entropy H_tar.

**Output:** shared actor parameters θ.

```text
 1: Initialize shared actor π_θ, twin critics Q_φ₁ and Q_φ₂,
    target parameters φ̄ₖ ← φₖ (k = 1, 2), log temperature ξ,
    α ← exp(ξ), and replay buffer D.
 2: for episode = 1, …, E do
 3:     Reset the environment; obtain state s and local observations o.
 4:     for t = 0, …, T − 1 do
 5:         Sample aᵢ ~ π_θ(· | oᵢ) independently for all UAVs.
 6:         Execute joint action a; update positions and A, B, Y, C.
 7:         Compute r = ρ + wτ + λJ_rob; observe s′, o′, and terminal flag d.
 8:         Store (s, o, a, r, s′, o′, d) in D.
 9:         if |D| ≥ L_start then
10:             Sample a minibatch of K joint transitions from D.
11:             Sample next joint actions from π_θ(· | o′);
                construct stopped-gradient soft Bellman targets y.
12:             Update φ₁ and φ₂ by minimizing their critic losses L_Q.
13:             Resample reparameterized joint actions from π_θ(· | o);
                update shared θ by minimizing L_π with critic weights fixed.
14:             Update ξ by minimizing L_temp with detached log probabilities;
                set α ← exp(ξ).
15:             Soft-update φ̄ₖ ← (1 − η)φ̄ₖ + ηφₖ, for k = 1, 2.
16:         end if
17:         Set s ← s′ and o ← o′; break if the episode ends.
18:     end for
19: end for
20: return θ.
```

The targets and losses follow Sections 4–5; entropy terms sum log probabilities over all UAVs. Here, `d` denotes task termination, not a collection-only truncation. The environment updates and robustness reward follow the problem model.

**Decentralized execution:** each UAV uses the trained shared actor and its local observation to generate movement actions; centralized critics and replay are used only during training.
