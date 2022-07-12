# Fluid-Particle-ALE
A code for the diploma thesis <em>Numerical simulations of interaction between fluid flow and rigid particles</em> at Charles University, Faculty of Mathematics and Physics.

Created by Jan Hrůza.

### Instalation
The [FEniCS](https://fenicsproject.org/) finite element library and [admesh](https://bitbucket.org/FaraJakub/admesh/src/admesh2.0/) mesh adaptation library created by Jakub Fara have to be installed to run the code.

### Running tohe code
To run e.g. the `ALE_2D_movement/ALE_solver.py`, create a initial mesh first, this is done using a `--mesh` parameter, the following series of commands will start the computation:
```
python3 ALE_2D_movement/ALE_solver.py --mesh
python3 ALE_2D_movement/ALE_solver.py
```
