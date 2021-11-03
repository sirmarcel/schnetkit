## `schnetkit`
### some tooling for `schnetpack`

**EXPERIMENTAL/IN DEVELOPMENT DO NOT USE**

This is an early draft of some infrastructure built around [`schnetpack`](https://github.com/atomistic-machine-learning/schnetpack). In particular, it allows:

- Saving/loading models from disk
- Initialising models from a `.yaml`-based description of the architecture
- Efficient `ase` calculator for `SchNet` models with support for a larger effective cutoff (i.e. avoiding to recompute the neighborlists on every step)

As a result of this, the `schnetkit` calculator is compatible with `FHI-vibes`. (Currently, this functionality is untested.)

Future versions will support:

- Restarteable training with automatic re-submission in `slurm`
- Training configurable with input files
- Efficient dataset class that pre-computes everything

This is a part of the infrastructure for the [`gknet`](https://marcel.science/gknet) project, and will be completed as the project nears publication.
