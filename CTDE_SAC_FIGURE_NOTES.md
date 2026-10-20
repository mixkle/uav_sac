# CTDE-SAC Framework Figure

The figure follows the two-panel visual organization of Figure 1 in *Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies* (PDF page 5). Its algorithmic content is adapted to the parameter-shared Gaussian actor and centralized twin-Q SAC formulation in `ENVIRONMENT MODELING.md`.

**Suggested caption:** Framework of parameter-shared multi-agent SAC for UAV deployment. Left: each UAV generates a movement action from its local observation using a shared Gaussian actor, and joint transitions are collected in a replay buffer. Right: centralized twin critics, target critics, the shared actor, and an adaptive entropy temperature are optimized from joint minibatches. Only the actor is required for decentralized movement decisions; user association follows the prescribed environment rule.

**Generation method:** Built-in image generation tool. The reference is used for layout and visual organization, not for diffusion-policy or distributional-critic algorithm content.

## Generation prompt

```text
Generate a new, clean academic FRAMEWORK FLOWCHART for "Parameter-Shared Multi-Agent SAC". White background, wide landscape, high resolution readable typography. Layout inspired by the CTDE Figure 1 of OMAD: TWO rounded panels side-by-side, pale blue LEFT "Decentralized Execution & Data Collection", pale peach RIGHT "Centralized Training", replay buffer cylinder at the seam. This is a simplified high-level algorithm flowchart, NOT a computational gradient graph. ALL SOLID ARROWS MEAN data/control flow. Do not draw any extra edges besides the ones explicitly listed below.

LEFT panel: top a single rounded blue "UAV Environment" box with small subtitle "Mobility -> A, B, Y, C updates" and second subtitle "Coverage + throughput + robustness". Include a tiny tasteful UAV-and-ground-users sketch inside only if ample space.
Bottom a stack of three blue rounded actor cards, front card text "UAV i: Shared Gaussian Actor", "Local observation o_i -> mean / std -> sample -> a_i". Beneath cards: "Shared parameters; independent samples".
One large blue arrow FROM environment left edge DOWN to actor left edge, arrowhead at ACTOR, labeled "Local observations".
One large blue arrow FROM actor right edge UP to environment right edge, arrowhead at ENVIRONMENT, labeled "Joint movement action".
At the middle seam (between panels, can overlap divider), a cylinder "Replay Buffer D". A solid arrow FROM environment to cylinder labelled "Joint transition". Near cylinder readable tuple "(s, o, a, r, s', o', d)". This line must not merge into the actor action loop.
A cylinder output arrow enters right panel's top block "Sample joint minibatch". Global state only enters replay and training; actors receive LOCAL observations.

RIGHT PANEL: Five rounded boxes in a single vertical column with ONLY downward short sequence arrows between them:
1. "Sample joint minibatch"
   subtitle "Global states, local observations, joint actions, rewards"
2. Orange "1  Update twin critics"
   subtitle "Q_phi1 and Q_phi2"
   second subtitle "Target critics + next local-policy actions; stopped Bellman target"
3. Blue "2  Update shared actor"
   subtitle "Fresh joint actions; value guidance from both critics"
   second subtitle "Joint entropy; critic weights fixed"
4. Green "3  Tune entropy temperature"
   subtitle "alpha = exp(xi); joint target entropy"
5. Orange "4  Soft-update target critics"
   subtitle "Target parameters <- (1 - eta) target + eta online"

A single dashed BLUE arrow from LEFT EDGE of right panel's "2 Update shared actor" box routed along the bottom margin to the bottom of the execution actor stack, arrowhead at actor, labelled "Synchronize shared parameters". This dashed arrow is the ONLY cross-panel parameter update arrow.
At bottom of right panel small note "Training only: twin critics, target critics, replay buffer".
Bottom page legend "Solid arrows: data / update sequence    Dashed arrow: parameter synchronization".
Do NOT show arrows between Q networks separately; the training boxes are stages, not a detailed computation graph. Do NOT add equations for losses or entropy targets. Do NOT use diffusion policy, KL loss, target actor, or distributional critic. Do not add alpha arrows looping around the panel.
All words must be correctly spelled; no illegible tiny labels. Use restrained blue/orange/green, very light fills, dark navy text, consistent arrowheads, abundant whitespace, no gradients or textures. The diagram should be suitable as a readable Figure 1 in an academic paper.
```
