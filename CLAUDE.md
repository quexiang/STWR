# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

STWR (Spatiotemporal Weighted Regression) is a Python package for calibrating geographically weighted regression (GWR), multiscale GWR (MGWR), and spatiotemporal weighted regression (STWR) models. It extends the `spglm` (sparse generalized linear modeling) framework with spatiotemporal kernels. Version 2.1.2, Python >3.4.

## Build / test / docs

```bash
# Install in editable mode with all extras
pip install -e .[tests,docs]

# Run tests (nose)
nosetests

# Build docs (Sphinx)
cd doc && make html
```

Test dependencies are `nose`, `nose-progressive`, `nose-exclude`, `coverage`, `coveralls`. Doc dependencies are `sphinx` and related plugins (see `requirements_docs.txt`). Core dependencies: `scipy`, `numpy`, `libpysal`, `spglm`, `pandas` (see `requirements.txt`). Optional: `numba` for JIT-compiled distance functions.

## Architecture

### Model class hierarchy

All models build on `spglm.glm.GLM` and results on `spglm.glm.GLMResults`:

- **`GWR`** (`gwr.py`) — standard geographically weighted regression. Supports Gaussian, Poisson, Binomial families via IWLS (`spglm.iwls`). Fit returns `GWRResults` (full) or `GWRResultsLite` (minimal diagnostics, used during bandwidth selection). Supports prediction at unsampled locations.
- **`MGWR`** (`gwr.py`) — multiscale GWR; each covariate gets its own bandwidth. Requires a pre-run `Sel_BW` selector. Fit uses iterative GAM backfitting (chunked computation to reduce memory) or a closed-form `exact_fit()`. Returns `MGWRResults`. Gaussian family only.
- **`STWR`** (`gwr.py`) — spatiotemporal weighted regression. Extends GWR into the temporal dimension by borrowing observations from past time stages. Uses a combined spatiotemporal kernel that weights by both spatial proximity and temporal value-change rate. Returns `STWRResults`.

### Kernel system (`kernels.py`)

Two parallel kernel systems:

1. **`Kernel`** (spatial-only) — used by GWR/MGWR. Computes distance via `local_cdist` (Euclidean or Haversine), then applies a kernel function (Gaussian, bisquare, exponential, etc.). Supports fixed (distance) and adaptive (k-nearest-neighbor) bandwidths.

2. **`_SpatiotemporalKernel`** (spatiotemporal) — used by STWR. Computes combined spatial-temporal distance matrices via `cspatiltemporaldist`, then applies weighted combination of spatial kernel and sigmoid temporal kernel via `spatialtemporalkernel_funcs`. The `alpha` parameter (0–1) balances spatial vs. temporal influence. The `sita` parameter controls how spatial bandwidth shrinks over time stages.

Kernel dispatch tables (`fk` for fixed, `ak` for adaptive) map kernel name strings to functions in `sel_bw.py`.

### Bandwidth selection (`sel_bw.py`, `search.py`)

- **`Sel_BW`** — selects bandwidth(s) for GWR or MGWR. Supports golden section search, equal interval search, or scipy optimizer. Criterion: AICc, AIC, BIC, or CV. For MGWR (`multi=True`), uses iterative backfitting via `multi_bw()` in `search.py`.
- **`Sel_Spt_BW`** — selects parameters for STWR: `alpha` (spatial/temporal balance), `sita` (spatial bandwidth slope over time), optimal number of time stages, and initial GWR bandwidth. Brute-force grid search over these parameters.

Search routines in `search.py`: `golden_section()` and `equal_interval()` are generic optimizers. `multi_bw()` implements the MGWR backfitting algorithm.

### Diagnostics (`diagnostics.py`)

Functions `get_AIC`, `get_AICc`, `get_BIC`, `get_CV` work on any model result object that has `.n`, `.tr_S`, `.llf`, `.y`, `.mu`, `.family` attributes.

### Summary output (`summary.py`)

`summaryModel` + `summaryGLM` + `summaryGWR`/`summaryMGWR`/`summarySTWR` compose the full printed summary for each model type. `summaryGLM` fits a global (non-spatial) GLM for comparison.

### Utilities (`utils.py`)

Matplotlib/geopandas visualization helpers: `shift_colormap`, `truncate_colormap`, `compare_surfaces` (side-by-side GWR vs MGWR parameter surface plots).

### Key external dependency: spglm

This package is tightly coupled to `spglm`. The `GLM` base class provides the IWLS fitting machinery (`spglm.iwls._compute_betas_gwr`, `spglm.iwls.iwls`), families (`Gaussian`, `Poisson`, `Binomial`), and `GLMResults` with its `@cache_readonly` pattern that all result classes inherit.
