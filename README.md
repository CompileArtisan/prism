### Where the data comes from

It uses **DeepMIMO**, an online catalog of simulated buildings ("scenarios," e.g. `i2_28b`, `i1_2p5`) where ray-tracing physics has already computed realistic signal propagation. Loading a scenario gives you:
- **positions**: (x, y, z) coordinates of receiver points
- **rss**: the true signal strength at each point (this is the "ground truth" you're trying to predict elsewhere)
- **env_feats**: features like distance-to-transmitter and a **LOS flag** (line-of-sight — whether a wall blocks the direct path)

Since real buildings have thousands of points, the notebook subsamples down to a manageable number.

### Turning points into a graph

To use graph-based AI models, the scattered points need to be connected into a network (a **graph**), where each point is a **node** and connections between nearby points are **edges**. Two ways to draw these edges:
- **k-NN graph**: connect each point to its *k* nearest neighbors (like "your 6 closest thermometers are your friends").
- **Delaunay graph**: a triangulation of the floor plan that connects points in a way that avoids long skinny slivers — more like a natural mesh.

Edges are also weighted by inverse distance (closer neighbors matter more), and weights are normalized so they behave like a proper weighted average.

### The prediction models

Four families of models compete against each other:

1. **GCN (Graph Convolutional Network)** — a neural network that updates each point's prediction by averaging information from its graph neighbors, using the edge weights above.
2. **GAT (Graph Attention Network)** — similar, but instead of fixed distance-based weights, it *learns* how much attention to pay to each neighbor (like deciding some neighbors are more "trustworthy" than others based on their features, not just distance).
3. **IDW (Inverse Distance Weighting)** — a simple classical baseline: predict a point's value as a distance-weighted average of nearby *known* values. No learning involved, just geometry.
4. **Kriging** — a classical geostatistics technique that models spatial correlation more rigorously (fits a "variogram" describing how similar nearby points tend to be) — the traditional gold-standard baseline this field compares against.
5. **Hybrid-GNN-IDW** — a clever combo: let IDW handle the smooth, easy part of the signal decay, and train a GNN to learn only the *leftover error* (the "residual") — things like multipath effects and LOS/NLOS transitions that IDW can't capture. This is often easier than learning everything from scratch.

### Preprocessing tricks (important, easy to miss)

- **FSPL (Free-Space Path Loss)** is subtracted from raw RSS before training. This is the theoretical signal drop-off with distance in open space — stripping it out removes scenario-specific absolute power baked into the data, leaving a cleaner "residual" for the model to learn.
- Features and residual RSS are **z-scored** (normalized to mean 0, std 1) — neural nets train much better on standardized numbers than raw physical units.
- Only `dist_to_tx` and `los_flag` are fed into the models — **not raw (x, y, z) position**. Position is only used to build the graph. Why? Because "this point is at coordinate (3.1, -0.4)" doesn't mean anything similar between two different buildings — so if you fed raw position to the model, it would just memorize "where the strong-signal spot is in this specific building" instead of learning transferable physics.

### The control flow — how everything runs together

1. **Setup**: install packages, pick a scenario from the live catalog.
2. **Load scenario**: get positions/RSS/features/frequency.
3. **Build graph(s)**: k-NN and/or Delaunay edges from positions.
4. **Split data**: randomly choose a small set of "labeled" (training) points and treat the rest as unlabeled ("test") points — simulating the real-world situation where you only have a few actual measurements.
5. **Normalize**: fit mean/std stats *only* on the training points (never peek at test data), transform both train and test.
6. **Train**: for each model, run a training loop (**epochs** = training passes) using **Adam** optimizer, with **early stopping** (stop once validation performance stops improving, to avoid overfitting) and **gradient clipping** (prevents unstable huge updates).
7. **Predict + denormalize**: convert the model's normalized output back into real dBm units (adding the FSPL curve back in).
8. **Score**: compute **RMSE** (root-mean-squared error — how far off predictions are on average) and **coverage accuracy** (did the model correctly say "covered" vs "dead zone" relative to a signal threshold?).
9. **Sweep**: repeat all of this across combinations of:
   - **density** (how many labeled points — sparse vs. dense),
   - **topology** (k-NN vs. Delaunay),
   - **model** (GCN, GAT, Hybrid, IDW, Kriging),
   - **multiple random seeds** (to report results as mean ± standard deviation, not a single lucky/unlucky run).
10. **Analyze**: pivot the results to explicitly quantify things like "does Delaunay beat k-NN?" rather than eyeballing a chart.
11. **Cross-scenario generalization**: train the model on Building A, then test it — with zero further training — on Building B, in both directions. This checks whether the model learned genuine "physics" (distance/LOS → signal) rather than memorizing one building's layout. A small **calibration** step (a handful of test-building points) can correct for absolute power-level offsets between buildings, but those calibration points are excluded from the final scored results to avoid leaking test answers into evaluation.
12. **Sparse-regime benchmark**: the final deliverable — specifically tests the hypothesis that GNNs beat classical methods most when you have very few measurements, since that's when their ability to share information across the graph gives them an edge over purely local methods like IDW.

# Result
i2_28b result — GCN beating IDW/Kriging by 15–23%

