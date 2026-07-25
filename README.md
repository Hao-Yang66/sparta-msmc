# SPARTA-MSMC

**Version 1.0**

An efficient particle solver for continuum-to-rarefied gas flows  
based on the Multiscale Simulation Monte Carlo method.

SPARTA-MSMC is a parallel particle simulation software package for
multiscale and nonequilibrium gas flows. It implements the Multiscale
Simulation Monte Carlo (MSMC) method for simulations across continuum,
transitional, and rarefied flow regimes.

The software supports Cartesian grid generation, particle transport,
gas-surface interactions, MSMC collision modeling, polyatomic
internal-energy relaxation, macroscopic sampling, surface diagnostics, and
MPI-based parallel computing.

SPARTA-MSMC is developed on the open-source SPARTA framework. The original
SPARTA copyright notices and third-party licenses are retained in the
corresponding source files and repository license materials.

## Quick Start

### 1. Download

Download the repository from GitHub with **Code > Download ZIP** and extract
it. Git users can instead copy the HTTPS address shown under **Code** and
clone the repository from a terminal. Enter the extracted or cloned
`sparta-msmc` directory before continuing.

### 2. Install build dependencies

On Ubuntu or Windows Subsystem for Linux (WSL), install the required compiler,
CMake, and Open MPI packages:

```bash
sudo apt update
sudo apt install -y build-essential cmake openmpi-bin libopenmpi-dev
```

### 3. Build

```bash
cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The compiled MPI executable is `build/src/spa_mpi`.

### 4. Run the bundled MSMC example

```bash
cd examples/msmc_circle
chmod +x run_mpi.sh
./run_mpi.sh
```

The example performs 2000 warm-up steps followed by 1000 sampling steps. Its
log, grid data, and surface data are written to `output/`.

### 5. Post-process

Open MATLAB in `examples/msmc_circle/post_process` and run:

```matlab
plot_flowfield
plot_surface
```

The scripts create `circle_msmc_temperature.png` and
`circle_msmc_surface.png` in the example's `output/` directory.

## Key Features

- MSMC simulation of continuum-to-rarefied gas flows
- Particle transport and collision modeling
- Translational, rotational, and vibrational energy relaxation
- Two-dimensional, axisymmetric, and three-dimensional simulations
- Cartesian grids with local refinement
- Gas-surface interaction and surface collision models
- Grid-based and surface-based statistical output
- Distributed-memory parallel computing with MPI

## Repository Structure

```text
sparta-msmc/
|-- cmake/       CMake build scripts, options, and presets
|-- data/        Species, collision, and geometry data
|-- doc/         Command reference and supporting documentation
|-- examples/    Self-contained MSMC example and input data
|-- python/      Python interface and helper files
|-- src/         Core C++ source code
|-- tools/       Pre-processing, conversion, and post-processing tools
|-- BUILD_CMAKE.md
|-- LICENSE
`-- README.md
```

## Requirements

A typical Linux or Windows Subsystem for Linux (WSL) build requires:

- A C++ compiler with C++11 support
- CMake 3.10 or later
- An MPI implementation such as Open MPI or MPICH
- GNU Make or another CMake-supported build tool

Optional libraries and accelerator packages can be enabled through the CMake
configuration options described in `BUILD_CMAKE.md`.

## Build

From the repository root, configure and compile SPARTA-MSMC with:

```bash
cmake -S cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The default MPI-enabled executable is:

```text
build/src/spa_mpi
```

The repository is named `sparta-msmc`; its MPI-enabled executable is named
`spa_mpi`.

To install the executable and libraries into a separate directory:

```bash
cmake -S cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j
cmake --install build
```

The installed executable is then available as:

```text
install/bin/spa_mpi
```

The traditional make-based build is also available from `src/`:

```bash
cd src
make mpi -j
```

This build produces `src/spa_mpi`.

## Run

SPARTA-MSMC is controlled by text input scripts. The bundled example can be
run from the repository root as follows:

```bash
cd examples/msmc_circle
./run_mpi.sh
```

At startup, the program identifies itself as:

```text
SPARTA-MSMC V1.0
An efficient particle solver for continuum-to-rarefied gas flows
Based on the Multiscale Simulation Monte Carlo method
```

The example uses four MPI processes by default. See
`examples/msmc_circle/README.md` for direct commands and configuration
options.

## MSMC Input Commands

The MSMC collision model is selected with the `collide msmc` command. A
typical configuration has the form:

```text
collide msmc flow ar.msmc mode msmc
collide_msmc_modify pr_num 0.6666667 time_ave 0.999 \
                     reset_wmax 0.99 interpolate yes
```

The bundled example uses this configuration with a 50 x 25 base grid and two
nested refinement levels.

## Output and Post-processing

Depending on the input script, a simulation can produce:

- Screen and log output describing initialization and runtime statistics
- Grid-based macroscopic flow-field data
- Surface pressure, shear stress, and energy-flux data
- Particle data for diagnostics
- Restart files for continuing a simulation

The bundled example includes MATLAB scripts that convert the final grid and
surface dumps into a temperature-field image and surface-distribution plots.
Additional files in `tools/` support other conversion and post-processing
workflows.

## Documentation

- `BUILD_CMAKE.md` describes CMake configuration and build options.
- `doc/` contains the command reference inherited from and extended beyond the
  underlying SPARTA framework.
- `examples/msmc_circle/` contains the complete reproducible MSMC example.

## License and Acknowledgments

SPARTA-MSMC contains software developed on the SPARTA framework and includes
third-party components. See `LICENSE` and the copyright notices in individual
source files for the applicable terms and acknowledgments.
