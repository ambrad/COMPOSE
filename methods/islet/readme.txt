This directory and the directory ../slmm contain the code used to generate the
results in the Islet 2D paper.

The program "cslunstab" demonstrates the unstable classical cubic interpolation
semi-Lagrangian instances.

The program "search" is used to find the Islet bases.

The directory "figures" contains the scripts used to generate data and figures.
The file "figures/figs.tex" contains the latex for the figures. Comments before
each figure explain how to generate the data and then the figure from these
data. Bash and hy scripts are those in the "figures" directory.

hy is a Lisp that compiles to Python AST. We used hy 0.18.0 with CPython 3.7.6
provided by Anaconda 3.

Programs need BLAS, LAPACK, and Kokkos. NetCDF is optional and was not used in
for the results in this paper. We used Kokkos version 3.1 in our build.

The code used to obtain performance data on Summit will be part of main E3SM
soon. The exact version used to generate the data is archived here:
    https://github.com/ambrad/E3SM/releases/tag/islet-2d-paper-summit-sl-gpu-timings

