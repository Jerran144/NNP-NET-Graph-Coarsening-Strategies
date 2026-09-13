# NNP-NET-Graph-Coarsening-Strategies

Codebase for Master's thesis: Enhancing NNP-NET by Refined Graph Coarsening Strategies.

This repository contains four new variants of the NNP-NET graph layout algorithm, developed as part of a Master's thesis to improve NNP-NET by refining its graph coarsening strategies.

## The Variants

The four variants explored in this repository are:

1. **NNP-NET - Adaptive Solar System Collapsing (ASSC)**
2. **NNP-NET - Edge Contraction (Extension)**
3. **NNP-NET - Pivot-Points Optimization**
4. **NNP-NET - Solar System Collapsing (Sun Promotion)**

Each variant's directory contains a full, self-contained codebase based on the original NNP-NET implementation. The standard build process, usage instructions, and most command-line arguments are identical to the original. Please refer to the `README.md` inside any of the variant directories for the base documentation.

However, there are a few new additions introduced across these variants:

### New Command Line Options

- `--ignore_weights`: Added to **all** variants. Instructs the algorithm to ignore edge weights.
- `-dsp`, `--dsp`, `--disable_stall_protection`: Added exclusively to the **ASSC** variant. Disables stall protection during layout generation.

### CSV Metrics Output

In addition to the standard graph layout output (`.vna` file), the variants now generate a `.csv` file containing detailed runtime metrics (e.g., embedding creation time, subgraph creation time, NNP-NET time, stress, neighbourhood preservation, etc.).

- When running on a single file, the metrics are saved to `{input_filename}_results.csv`.
- When running on a directory of graphs, the metrics are saved to `results.csv` in the specified output path.

---

*For all other instructions, including how to build and standard usage, please see the `README.md` located inside each variant's folder.*
=======

# 
