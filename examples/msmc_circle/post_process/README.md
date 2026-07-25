# Post-processing

Run the following MATLAB scripts after completing the example:

- `plot_flowfield.m` reads `../output/circle_msmc.grid.1000.dat` and writes
  `../output/circle_msmc_temperature.png`.
- `plot_surface.m` reads `../output/circle_msmc.surf.1000.dat` and writes
  `../output/circle_msmc_surface.png`. The figure contains pressure, the two
  shear-force components, and total energy flux along the cylinder surface.

The grid dump contains the following columns after its nine-line header:

1. cell ID
2. cell-center x coordinate
3. cell-center y coordinate
4. cell volume
5. number density
6. x velocity
7. y velocity
8. translational temperature
9. pressure

The surface dump contains surface ID, pressure, x and y shear-force
components, and total energy flux after its nine-line header. Surface IDs are
mapped to angles using `../data.circle`.
