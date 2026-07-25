#!/usr/bin/env bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../.." && pwd)"

mpi_procs="${MPI_PROCS:-4}"
sparta_exe="${SPARTA_EXE:-${repo_root}/build/src/spa_mpi}"

if [[ ! -x "${sparta_exe}" ]]; then
  echo "Error: spa_mpi was not found or is not executable: ${sparta_exe}" >&2
  echo "Build the program first or set SPARTA_EXE=/path/to/spa_mpi." >&2
  exit 1
fi

mkdir -p "${case_dir}/output"
cd "${case_dir}"

echo "Running SPARTA-MSMC V1.0 with ${mpi_procs} MPI processes"
echo "Executable: ${sparta_exe}"
mpirun -np "${mpi_procs}" "${sparta_exe}" -in in.circle -log output/log.sparta
