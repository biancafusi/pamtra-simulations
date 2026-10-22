# project-pamtra

Simulating the TROPOS MIRA-35 cloud radar (Invercargill, New Zealand) from ICON model meteograms using PAMTRA.

## Overview

This project provides a toolchain that:

1. **Reads ICON meteogram data** (NetCDF format) containing atmospheric profiles.
2. **Simulates the MIRA-35 cloud radar** (35.5 GHz) operated by the TROPOS group in Invercargill, New Zealand, using the PAMTRA radiative transfer model.
3. **Visualizes the simulated radar moments** (reflectivity, Doppler velocity, spectral width, and linear depolarization ratio) as time-height cross-sections.

The goal is to produce a modular Python tool that separates these three stages: data validation, PAMTRA simulation, and visualization.

## Current Files

| File | Description |
|------|-------------|
| `icon2pamtra_mira.py` | Converts ICON meteogram data to MIRA-35 PAMTRA simulations. Supports 1-moment and 2-moment microphysics schemes. |
| `pamtra_earthcare_track.py` | PAMTRA simulation script for EarthCARE satellite tracks. |
| `icon2PAMTRA_1mm_v2.ipynb` | Jupyter notebook with the original ICON-to-PAMTRA conversion workflow. |
| `mira35_quicklooks.ipynb` | Generates daily quicklook plots of simulated radar moments (Z, V, SW, LDR). |
| `35GHzcloudradarMira.md` | Configuration notes for the MIRA-35 radar parameters in PAMTRA. |

## Dependencies

This project depends on [PAMTRA](https://github.com/igmk/pamtra) (Passive and Active Microwave TRAnsfer) and its Python interface `pyPamtra`. Follow the [PAMTRA installation guide](https://pamtra.readthedocs.io/en/latest/installation.html) to set up the environment.

Additional Python packages used:

- `numpy`
- `scipy`
- `pandas`
- `xarray`
- `netCDF4`
- `matplotlib`
- `psutil`
- `jupyter` (for notebooks)

A quick way to install the Python dependencies is:

```bash
pip install numpy scipy pandas xarray netCDF4 matplotlib psutil jupyter
```

Then install PAMTRA itself:

```bash
pip install pamtra
```

## Usage

The workflow is still under development. The intended structure is:

1. **Validate** — check that ICON meteogram files are present and well-formed.
2. **Simulate** — run PAMTRA to generate MIRA-35 radar moments from the meteogram data.
3. **Visualize** — produce quicklook plots of the simulated radar moments.

For now, run `icon2pamtra_mira.py` to perform the full ICON-to-PAMTRA conversion, then use `mira35_quicklooks.ipynb` to visualize the results.

## License

This project is part of academic research conducted at the University of Hamburg (TROPOS group).

## References

- Mech, M., et al. (2020). "PAMTRA 1.0: the Passive and Active Microwave radiative TRAnsfer tool for simulating radiometer and radar measurements of the cloudy atmosphere." *Geoscientific Model Development*, 13(10), 4229-4254. [DOI: 10.5194/gmd-13-4229-2020](https://gmd.copernicus.org/articles/13/4229/2020/)
- [PAMTRA Documentation](https://pamtra.readthedocs.io/)
- [TROPOS MIRA-35 Quicklooks](https://rsd2.tropos.de/quicklooks/dataBrowser3.html?site=INV)

## Contact

For questions or collaboration, please reach out to the author, Bianca Fusinato.
# pamtra-simulations
