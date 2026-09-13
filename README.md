# Enhancing NNP-NET by Refined Graph Coarsening Strategies

This repository contains the codebase for the Master's thesis of Jeroen van Houten (Utrecht University). It extends the original [NNP-NET](https://github.com/IlanHartskeerl/NNP-NET) framework by implementing and evaluating four graph coarsening/subgraph extraction strategies to improve the quality and efficiency of NNP-NET.

## Repository Structure

This repository contains four separate, self-contained iterations of the NNP-NET codebase. Each folder corresponds to a specific subgraph extraction strategy discussed in the thesis:

*   **`adaptive_solar_system_collapsing/`**: Implementation of adaptive solar system collapsing (ASSC).
*   **`solar_system_collapsing_sun_promotion/`**: Implementation of solar system collapsing (sun promotion variant).
*   **`edge_contraction/`**: Implementation of edge contraction (and extension).
*   **`pivot_points_optimization/`**: Implementation of pivot-points pptimization.

**Note:** To test a specific method, you must build and run the executable from within its respective directory.

---

## Building the Code

Each of the four directories can be compiled independently using CMake. Ensure you have a C++ compiler, Python, and TensorFlow installed.

To build a specific iteration, navigate into its folder and run the standard build sequence:

```bash
cd <chosen_method_folder>
mkdir build
cd build
cmake ../
make
