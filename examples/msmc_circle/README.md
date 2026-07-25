# MSMC Hypersonic Cylinder Example

This directory contains one self-contained SPARTA-MSMC V1.0 example. It
simulates hypersonic argon flow over a two-dimensional half-cylinder using the
Multiscale Simulation Monte Carlo method.

## Case definition

- Base grid: 50 x 25 cells
- Grid refinement: two nested 2 x 2 refinement levels
- Free-stream temperature: 217.5 K
- Free-stream velocity: 4121.26 m/s
- Cylinder radius: 0.04 m
- Cylinder-wall temperature: 1000 K
- Warm-up stage: 2000 steps
- Sampling stage: 1000 steps
- Total: 3000 steps

The files `in.circle`, `ar.species`, `ar.msmc`, and `data.circle` are all kept
in this directory so the case does not depend on files elsewhere in the
repository.

## Run

Build `spa_mpi` from the repository root first. Then run this example on four
MPI processes:

```bash
cd examples/msmc_circle
chmod +x run_mpi.sh
./run_mpi.sh
```

The process count and executable path can be changed without editing the
script:

```bash
MPI_PROCS=8 SPARTA_EXE=/path/to/spa_mpi ./run_mpi.sh
```

The equivalent direct command is:

```bash
mkdir -p output
mpirun -np 4 ../../build/src/spa_mpi -in in.circle \
  -log output/log.sparta
```

## Output

Runtime information is printed in the terminal and saved in
`output/log.sparta`. During the sampling stage, the program writes:

- `circle_msmc.grid.1000.dat`: cell-averaged coordinates, volume, number
  density, velocity, temperature, and pressure.
- `circle_msmc.surf.1000.dat`: surface-averaged pressure, shear-force
  components, and total energy flux.

Generated files are ignored by Git. Run `post_process/plot_flowfield.m` and
`post_process/plot_surface.m` in MATLAB after the simulation to visualize the
final temperature field and surface-averaged quantities.
