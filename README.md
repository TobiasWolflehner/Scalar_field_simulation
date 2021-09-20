# Scalar_field_simulation
A simulation of interacting scalar fields in 1+1 dimensions implemented in python.
This simulation was coded for a bacchelor's thesis "Untersuchung des Einfluss des Koordinatensystems in einer numerischen Simulation reeller Skalarfelder in Lichtkegel- und Raum-Zeit Koordinaten", which was written in German.
For accessibility the simulation was coded and annotated entirely in English.

How to use (also found in the simulation code as is):

There is no UI for this simulation. Parameters have are currently hard coded
and included as global variables. How to set these parameters is explained below.
Install the needed libraries such as numpy, matplotlib and also the modified scipy
library from "https://github.com/TobiasWolflehner/scipy". If the official scipy
library is used, slight deviations occur.

First the parameters of the simulation are chosen; two coordinatesystems are
compared, lightcone coordinates and normal time-space Minkowski coordinates.
These parameters include the lenght of the arrays, which governs with the
Broadening parameter the discreteness of the lattice. This Broadening governs
the broadness of the Gaussian, which is used as a starting condition for the 
simulation. x0, x1 refers to normal coordinates, while p and n refers to x+
and x- in lightcone coordinates.

For ease of use, modify only Lenght_x0 (this needs to be uneven; this way the 
origin is always included in the lattice), BroadeningQuotient as well as the 
InteractionQuotient to desired values. If unclear, refer to the source thesis, 
the additional parameters serve to fine tune the algorithm, but do not usually 
have to be modified so long as the code is working.

The standard output of this simulation is a graph of the difference of the two 
simulations in lightcone and normal coordinates. A heatmap or a graph as 2D
slices is available and can the mode can be chosen with the Graph_as_Heatmap 
parameter.
