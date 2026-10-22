#!/usr/bin/env python3
"""
icon2pamtra_mira_v3.py
=====================

Converts ICON model meteogram data into MIRA-35 radar (35.5 GHz) simulations
via PAMTRA.

Version 3 improvements over v2:
- Descriptor files aligned with pamtra_earthcare_track.py (gscp2 for 1-moment,
  gscp4 for 2-moment), including hail (QH/NH) for the 2-moment scheme.
- Detailed documentation in run_pamtra_daily() with PAMTRA references.
- Hail (QH/NH) now included in all processing flags for the 2-moment scheme.
- Descriptor file selection simplified: only depends on the microphysics scheme.

Supports both 1-moment and 2-moment ICON microphysics schemes.
Supports 4 pre-processing modes: raw, total, subgrid, convective.

Usage:
    python icon2pamtra_mira_v3.py

Or, in a Jupyter Notebook:
    %run icon2pamtra_mira_v3.py

Author: Adapted from icon2pamtra_mira_v2.py and pamtra_earthcare_track.py
Date: 2025

References:
    - PAMTRA Documentation: https://pamtra.readthedocs.io
    - PAMTRA Descriptor File Structure:
      https://pamtra.readthedocs.io/en/latest/descriptor.html
    - PAMTRA Settings/Nml-File:
      https://pamtra.readthedocs.io/en/latest/settings.html
    - PAMTRA Tutorial: https://atmos.meteo.uni-koeln.de/~mech/pamtra.html
    - Geer et al. (2007): "The retrieval of humidities, cloud and
      precipitation information from radiosondes", Q.J.R. Meteorol. Soc.
    - Cooper (1986): "The interpretation of the variation in the
      concentration of ice crystals in clouds", Q.J.R. Meteorol. Soc.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import xarray as xr
import pyPamtra
import psutil

# Get current process for memory monitoring
process = psutil.Process(os.getpid())


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logger(output_dir):
    """
    Configura o logger para escrever em um arquivo de log no diretório de saída.

    Parameters
    ----------
    output_dir : str
        Diretório onde o arquivo de log será salvo.

    Returns
    -------
    logging.Logger
        Instância do logger configurado.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "pamtra_processing.log")

    logger = logging.getLogger("pamtra_processor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# INITIAL CONFIGURATION
# ==============================================================================

# Input file (ICON meteogram)
INPUT_FILE = "MTGRAM_INV_patch001.nc"

# Processing period (date filtering)
START_DATE = "2025-08-28"
END_DATE = "2025-10-01"

# Station of interest (index in the stations array)
STATION_INDEX = 0

# Input and output directories
INPUT_DIR = "/work/bb1358/fusinato/TROPOS/shared_files_project_inv/AUG_SEP_2025_10km/"
OUTPUT_DIR = "/home/b/b383757/scripts/pamtra/MIRA_flag_total_10km"

# Processing flag: "raw", "total", "subgrid", "convective"
#   - "raw":        Raw ICON data, no corrections
#   - "total":      Subgrid + convective effects
#   - "subgrid":    Subgrid-scale contributions only
#   - "convective": Convective effects only
PROCESSING_FLAG = "total"

# Microphysics scheme: "1mom", "2mom", or "auto"
# "auto" detects based on variables available in the meteogram
MICROPHYSICS = "1mom"

# Number of cores for parallel processing
CORES = 10


def print_memory_usage(label="", logger=None):
    """
    Prints current memory usage.

    Parameters
    ----------
    label : str
        Label prefixo para a mensagem.
    logger : logging.Logger, optional
        Se fornecido, escreve no log em vez de imprimir.
    """
    mem_info = process.memory_info()
    message = f"  {label} RSS: {mem_info.rss / (1024 * 1024):.2f} MB, " \
              f"VMS: {mem_info.vms / (1024 * 1024):.2f} MB"
    if logger:
        logger.info(message)
    else:
        print(message)


# ==============================================================================
# Based on pamtra_earthcare_track.py
# ==============================================================================

def nicooper(temp):
    """
    Ice number concentration using Cooper (1986).

    Parameters
    ----------
    temp : numpy.ndarray
        Temperature (K).

    Returns
    -------
    numpy.ndarray
        Ice number concentration (m^-3).
    """
    ni = 5.0E+0 * np.exp(0.304 * (273.15 - temp))
    return ni


def limit_number_density(q, n, x_min, x_max):
    """
    Limits the mean particle mass to [x_min, x_max] by adjusting the number density n.

    Parameters
    ----------
    q : numpy.ndarray
        Mass mixing ratio (kg/kg).
    n : numpy.ndarray
        Number concentration (m^-3).
    x_min : float
        Minimum particle mass (kg).
    x_max : float
        Maximum particle mass (kg).

    Returns
    -------
    numpy.ndarray
        Adjusted number concentration (m^-3).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.where((n > 1.0) & (q > 1e-12), q / n, 0.0)
        x = np.clip(x, x_min, x_max)
        n_limited = np.where(x > 0.0, q / x, 0.0)
    return n_limited


# ICON particle geometry parameters (2-moment scheme)
# Used to derive number concentrations in the 1-moment scheme
# Values from pamtra_earthcare_track.py
cloud_x_min = 4.20e-15   # minimum mass (D = 2 um)
cloud_x_max = 2.60e-10   # maximum mass (D = 80 um)
rain_x_min = 2.60e-10    # minimum mass (~80 um)
rain_x_max = 6.50e-5     # maximum mass (~5 mm)
ice_x_min = 1.00e-12     # minimum mass (~200 um)
ice_x_max = 1.00e-5      # maximum mass
snow_x_min = 1.00e-10    # minimum mass
snow_x_max = 2.00e-5     # maximum mass
graupel_x_min = 4.19e-9  # minimum mass
graupel_x_max = 5.30e-4  # maximum mass
hail_x_min = 2.60e-9     # minimum mass
hail_x_max = 5.00e-3     # maximum mass


def get_meteog_variables(meteog, station_index):
    """
    Extracts all 3D variables for a specific station.

    ICON stores variable names as bytes in var_name.
    Each variable is identified by an index (nvars).

    Parameters
    ----------
    meteog : xarray.Dataset
        ICON meteogram dataset.
    station_index : int
        Station index in the stations array.

    Returns
    -------
    dict
        Dictionary with variable name as key and DataArray as value.
    """
    variables = {}

    for i, name in enumerate(meteog["var_name"].values):
        # Convert bytes -> string
        name = name.decode().strip()

        # Extract the variable for the selected station
        variables[name] = meteog["values"].isel(
            nvars=i,
            nstations=station_index
        )

    return variables


def get_times(meteog):
    """
    Converts ICON dates to Unix timestamp (seconds since 1970-01-01).

    ICON format is: YYYYMMDDTHHMMSSZ
    Example: 20240328T000000Z = March 28, 2024, 00:00:00 UTC

    Parameters
    ----------
    meteog : xarray.Dataset
        ICON meteogram dataset.

    Returns
    -------
    numpy.ndarray
        Array of timestamps in seconds.
    """
    dates = meteog["date"].values

    # Convert bytes -> strings
    dates = np.array([
        d.decode().strip() if isinstance(d, bytes) else str(d).strip()
        for d in dates
    ])

    # Convert YYYYMMDDTHHMMSSZ -> YYYY-MM-DDTHH:MM:SS
    dates = np.array([
        f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{d[9:11]}:{d[11:13]}:{d[13:15]}"
        for d in dates
    ])

    # Convert to datetime64[ns] and then to Unix timestamp
    dates = dates.astype("datetime64[ns]")
    times = dates.astype(np.int64).astype(np.float64) * 1e-9

    return times


def get_available_variables(meteog):
    """
    Lists all variables available in the meteogram.

    Parameters
    ----------
    meteog : xarray.Dataset
        ICON meteogram dataset.

    Returns
    -------
    list
        List of variable names.
    """
    var_names = []
    for i, name in enumerate(meteog["var_name"].values):
        var_names.append(name.decode().strip())
    return var_names


def has_2moment_vars(var_names):
    """
    Checks if the meteogram contains number concentration variables
    (indicative of a 2-moment scheme).

    Parameters
    ----------
    var_names : list
        List of variable names.

    Returns
    -------
    bool
        True if 2-moment variables are present.
    """
    # Typical 2-moment number concentration variables
    n_vars = ["NC", "NR", "NI", "NS", "NG", "NH"]
    return any(nv in var_names for nv in n_vars)


# ==============================================================================
# DESCRIPTOR FILES
# Based on pamtra_earthcare_track.py (descriptorFile_gscp2 and gscp4)
#
# PAMTRA descriptor file structure (20 columns per hydrometeor):
#   [0]  hydro_name    - Name/identifier of the hydrometeor class
#   [1]  as_ratio       - Aspect ratio (fixed value or -99 for variable)
#   [2]  liq_ice        - Phase: 1 = liquid, -1 = ice
#   [3]  rho_ms         - Density coefficient for mass-size relation
#   [4]  a_ms           - Prefactor in mass-diameter relation: m = a_ms * D^b_ms
#   [5]  b_ms           - Exponent in mass-diameter relation: m = a_ms * D^b_ms
#   [6]  alpha_as       - Aspect ratio coefficient (Dpar = alpha_as * D^beta_as)
#   [7]  beta_as        - Aspect ratio exponent
#   [8]  moment_in      - Input moment type:
#                        3  = 1-moment (mass only, N0 from descriptor)
#                        13 = 2-moment (mass + number concentration from profile)
#   [9]  nbin           - Number of bins for the particle size distribution
#   [10] dist_name      - PSD type: 'mono', 'exp', 'mgamma', 'exp_cosmo_snow', etc.
#   [11] p_1            - PSD parameter 1 (e.g. N0 for 'exp', nu for 'mgamma')
#   [12] p_2            - PSD parameter 2 (e.g. lambda for 'exp', mu for 'mgamma')
#   [13] p_3            - PSD parameter 3
#   [14] p_4            - PSD parameter 4
#   [15] d_1            - Minimum particle diameter (m)
#   [16] d_2            - Maximum particle diameter (m)
#   [17] scat_name      - Scattering model: 'mie-sphere', 'ssrg-rt3', etc.
#   [18] vel_size_mod   - Terminal velocity model: 'corPowerLaw_A_B', etc.
#   [19] canting        - Canting angle model (-99 = disabled)
#
# Reference: https://pamtra.readthedocs.io/en/latest/descriptor.html
# ==============================================================================


def get_descriptor_1moment():
    """
    Returns the descriptor file for the ICON 1-moment scheme (inwp_gsp = 2).
    Aligned with descriptorFile_gscp2 from pamtra_earthcare_track.py.
    5 hydrometeors: cloud, ice, rain, snow, graupel.
    No number concentrations (moment_in = 3).

    Returns
    -------
    numpy.ndarray
        Structured array with hydrometeor properties.
    """
    return np.array([
        # cloud liquid water (1-moment mass)
        ('cwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1,
         'mono', -99.0, -99.0, -99.0, -99.0, 2.0e-5, -99.0,
         'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
        # cloud ice (1-moment mass)
        ('iwc_q', 0.2, -1, -99.0, 130.0, 3.0, 0.684, 2.0, 3, 1,
         'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0, -99.0, -99.0,
         'ssrg-rt3_0.18_0.89_2.06_0.08', 'corPowerLaw_10.0_0.5', -99.0),
        # rain (1-moment mass)
        ('rwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 100,
         'exp', -99.0, -99.0, 1.0e6, -99.0, 1.2e-4, 6.0e-3,
         'mie-sphere', 'corPowerLaw_130.0_0.5', -99.0),
        # snow (1-moment mass)
        ('swc_q', 0.6, -1, -99.0, 0.038, 2.0, 0.3971, 1.88, 3, 100,
         'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, 5.1e-11, 1.0e-2,
         'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0),
        # graupel with N0g=4e6
        ('gwc_q', 1.0, -1, -99.0, 169.6, 3.1, 0.5, 2.5, 3, 100,
         'exp', -99.0, -99.0, 4.0e6, -99.0, 8.0e-11, 5.0e-3,
         'ssrg-rt3_0.30_1.10_1.80_0.05', 'corPowerLaw_5.0_0.20', -99.0),
    ])


def get_descriptor_2moment():
    """
    Returns the descriptor file for the ICON 2-moment scheme (inwp_gsp = 4).
    Aligned with descriptorFile_gscp4 from pamtra_earthcare_track.py.
    6 hydrometeors: cloud, ice, rain, snow, graupel, hail.
    With number concentrations (moment_in = 13 indicates 2-moment).

    For the 2-moment scheme, PAMTRA uses the generalized gamma PSD (mgamma):
        f(D) = N0 * D^nu * exp(-lambda * D^mu)
    where nu and mu are passed from the profile (hydro_n) and the descriptor.

    Hail (hwc_q) is included, matching the ICON 2-moment microphysics scheme.

    Returns
    -------
    numpy.ndarray
        Structured array with hydrometeor properties.
    """
    return np.array([
        # cloud liquid water (2-moment)
        ('cwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 13, 100,
         'mgamma', -99.0, -99.0, 1.0, 1.0, 2.0e-6, 8.0e-5,
         'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0),
        # cloud ice (2-moment)
        ('iwc_q', 1.0, -1, -99.0, 130.0, 3.0, 0.684, 2.0, 13, 100,
         'mgamma', -99.0, -99.0, 1.564, 0.8547, 1.744e-5, 9.369e-3,
         'ssrg-rt3', 'corPowerLaw_30.606_0.5533', -99.0),
        # rain (2-moment)
        ('rwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 13, 100,
         'mgamma', -99.0, -99.0, 2.0, 1.0, 0.00012, 8.2e-3,
         'mie-sphere', 'corPowerLaw_494.74_0.7031', -99.0),
        # snow (2-moment)
        ('swc_q', 0.6, -1, -99.0, 0.038, 2.0, 0.3971, 1.88, 13, 100,
         'mgamma', -99.0, -99.0, 1.0, 1.0, 5.13e-5, 2.294e-2,
         'ssrg-rt3', 'corPowerLaw_5.511054_0.25', -99.0),
        # graupel (2-moment)
        ('gwc_q', 1.0, -1, -99.0, 500.86, 3.18, -99.0, -99.0, 13, 100,
         'mgamma', -99.0, -99.0, 5.37, 1.06, 2.11e-4, 1.3e-2,
         'mie-sphere', 'corPowerLaw_406.67_0.85', -99.0),
        # hail (2-moment) — NEW in v3
        ('hwc_q', 1.0, -1, -99.0, -99.0, -99.0, -99.0, -99.0, 13, 100,
         'mgamma', -99.0, -99.0, 1.0, 0.333333, 500.0e-6, 8.0e-3,
         'mie-sphere', 'corPowerLaw_39.3_0.166667', -99.0),
    ])


def process_convective_precipitation(rain_con, snow_con, temp, rho,
                                     use_subgrid_convection):
    """
    Converts convective precipitation rates to mixing ratios and
    number concentrations using the approach by Geer et al. (2007).

    Parameters
    ----------
    rain_con : numpy.ndarray
        Convective rain precipitation rate (kg m⁻² s⁻¹), half levels.
    snow_con : numpy.ndarray
        Convective snow precipitation rate (kg m⁻² s⁻¹), half levels.
    temp : numpy.ndarray
        Temperature (K), full levels.
    rho : numpy.ndarray
        Air density (kg/m³), full levels.
    use_subgrid_convection : bool
        If True, applies the convective conversion.

    Returns
    -------
    tuple
        (qrcon, qscon, nrcon, nscon) — mixing ratios and number concentrations
        of convective rain and snow.
    """
    if not use_subgrid_convection:
        shape = temp.shape
        return (np.zeros(shape), np.zeros(shape),
                np.zeros(shape), np.zeros(shape))

    # Parameters from Geer et al. (2007)
    rho_w = 1000.0      # liquid water density (kg/m³)
    t0 = 273.16         # melting point (K)
    n0r = 8e6           # exponential rain PSD intercept
    ascon = 0.069       # prefactor in m = a*D²
    n0s1 = 13.5 * 5.65e5  # snow PSD parameter (Field et al.)
    n0s2 = 0.107        # snow PSD parameter (Field et al.)

    # Conversion coefficients
    a_rain = 20.89 / 3600.0   # mm h⁻¹ -> kg m⁻² s⁻¹
    a_snow = 29.51 / 3600.0
    b_rain = 1.15
    b_snow = 1.10

    # 1. Precipitation rate -> Volumetric concentration (g m⁻³)
    amount_rain_con_hf = (rain_con / a_rain) ** (1.0 / b_rain)
    amount_snow_con_hf = (snow_con / a_snow) ** (1.0 / b_snow)

    # 2. Half levels -> Full levels (arithmetic mean)
    amount_rain_con_f = 0.5 * (
        amount_rain_con_hf[:, :-1] + amount_rain_con_hf[:, 1:]
    )
    amount_snow_con_f = 0.5 * (
        amount_snow_con_hf[:, :-1] + amount_snow_con_hf[:, 1:]
    )

    # 3. g m⁻³ -> kg m⁻³
    amount_rain_con_f = amount_rain_con_f * 1e-3
    amount_snow_con_f = amount_snow_con_f * 1e-3

    # 4. Concentration -> Mixing ratio (kg/kg)
    rain_mr = amount_rain_con_f / rho
    snow_mr = amount_snow_con_f / rho

    # 5. Number concentrations from PSD
    qrcon = rain_mr
    qscon = snow_mr

    # Snow number concentration (temperature-dependent)
    n0s = n0s1 * np.exp(-n0s2 * np.clip(temp - t0, -40.0, 0.0))
    n0s = np.clip(n0s, 1.0e6, 1.0e9)

    nscon = n0s ** (2 / 3) * (rho * qscon / (2.0 * ascon)) ** (1 / 3) / rho
    nrcon = n0r ** (3 / 4) * (rho * qrcon / (np.pi * rho_w)) ** (1 / 4) / rho

    return qrcon, qscon, nrcon, nscon


def run_pamtra_daily(times_batch, press_batch, temp_batch, rh_batch,
                     uv_batch, w_batch, hydro_q, hydro_n,
                     zfull_pamtra, lat, lon, descriptorFile,
                     output_file, cores=4, logger=None):
    """
    Runs PAMTRA for a batch of data (one day).

    This function creates a PAMTRA atmospheric profile from ICON model data,
    configures the MIRA-35 radar (35.5 GHz), and runs the simulation.

    STEP-BY-STEP EXPLANATION:

    1. PREPARE SPATIAL COORDINATES
       PAMTRA expects height profiles ordered from bottom to top (surface ->
       upper atmosphere). The ICON meteogram provides heights in full levels.
       We tile the height array for each time step in the batch.

       Reference: PAMTRA createProfile expects `hgt` with shape (ntime, nlev)
       sorted ascending (bottom-up).
       https://pamtra.readthedocs.io/en/latest/profile.html

    2. CREATE THE PAMTRA OBJECT
       `pyPamtra.pyPamtra()` initializes an empty PAMTRA instance with default
       settings. The object has two main dictionaries:
       - `pam.nmlSet`: Namelist settings (radar mode, frequencies, etc.)
       - `pam.set`: Low-level settings (verbosity, etc.)
       and a `pam.p` dictionary for profile-level variables.

       Reference: https://pamtra.readthedocs.io/en/latest/quickstart.html

    3. ADD HYDROMETEORS FROM THE DESCRIPTOR FILE
       Each row in the descriptor file defines one hydrometeor class with its
       physical properties (PSD, mass-diameter relation, scattering model, etc.).
       `pam.df.addHydrometeor()` registers each hydrometeor with PAMTRA.

       The descriptor file structure (20 columns) is documented at:
       https://pamtra.readthedocs.io/en/latest/descriptor.html

       Key columns:
       - `moment_in` (col 8): 3 = 1-moment (mass only), 13 = 2-moment (mass + number)
       - `dist_name` (col 10): PSD type ('mono', 'exp', 'mgamma', etc.)
       - `scat_name` (col 17): Scattering model ('mie-sphere', 'ssrg-rt3', etc.)
       - `vel_size_mod` (col 18): Terminal velocity model

    4. CREATE THE ATMOSPHERIC PROFILE
       `pam.createProfile()` builds the atmospheric state from the input arrays.
       Required parameters:
       - `hgt`: Height in meters (bottom-up), shape (ntime, nlev)
       - `press`: Pressure in Pa, shape (ntime, nlev)
       - `temp`: Temperature in K, shape (ntime, nlev)
       - `relhum`: Relative humidity in %, shape (ntime, nlev)

       Optional but critical for radar simulation:
       - `hydro_q`: Hydrometeor mixing ratios (kg/kg), shape (ntime, nlev, n_hydro)
         Must match the order of hydrometeors in the descriptor file.
       - `hydro_n`: Number concentrations (m^-3), shape (ntime, nlev, n_hydro)
         Used when moment_in = 13 (2-moment scheme).
         For 1-moment, set to zeros or small values.
       - `wind_uv`: Horizontal wind speed (m/s), affects Doppler broadening
       - `wind_w`: Vertical wind speed (m/s), affects Doppler shift
       - `timestamp`: Unix timestamps (seconds), for output metadata
       - `lat` / `lon`: Station coordinates, for output metadata

       Reference: https://pamtra.readthedocs.io/en/latest/profile.html

    5. CONFIGURE THE RADAR SIMULATION
       All radar settings are stored in `pam.nmlSet`. Key settings:

       - `passive` / `active`: Enable passive (radiometer) or active (radar) mode.
         For radar-only simulation: passive=False, active=True.

       - `radar_mode`: Simulation mode:
         * 'simple': Only computes Ze by integrating over diameter.
         * 'spectrum': Simulates the full Doppler spectrum + all moments.
                       Slower and more memory-intensive.
         * 'moments': Identical to 'spectrum' but discards the full spectrum
                      after computing moments, saving memory.
         We use 'moments' for Ze, MDV, SW, Skewness, Kurtosis.

       - `radar_fwhr_beamwidth_deg`: Radar beamwidth (degrees). Affects spectral
         broadening estimation. MIRA-35: 0.5°.

       - `radar_integration_time`: Integration time (seconds). Affects the
         velocity resolution and noise level. MIRA-35: 2.8 s.

       - `radar_max_v` / `radar_min_v`: Maximum/minimum Nyquist velocity (m/s).
         Defines the unambiguous velocity range.
         MIRA-35: ±6.255 m/s.

       - `radar_aliasing_nyquist_interv`: Number of Nyquist intervals.
         Total unambiguous velocity = radar_max_v * radar_aliasing_nyquist_interv.
         Set to 4 to cover ~25 m/s (6.255 * 4 = 25.02 m/s).

       - `radar_nfft`: FFT size for spectrum computation. Higher = better
         resolution but slower. Default: 512.

       - `radar_no_ave`: Number of spectral averages. More averages = smoother
         spectrum but longer integration time. Default: 30.

       - `radar_pnoise0`: Receiver noise level (dB). Lower = more sensitive.
         MIRA-35: -56 dB.

       - `radar_polarisation`: Radar polarization mode.
         'NN' = non-polarized (mono-polar radar like MIRA-35).
         'HH,VV,HV' = dual-polarization (for ZDR, LDR, etc.).

       - `radar_airmotion`: Include air motion effects on Doppler spectrum.
         Important for accurate MDV and spectral width.

       - `radar_airmotion_model`: Model for air motion turbulence.
         'constant': Uses a constant EDR value from pam.p['turb_edr'].

       Reference: https://pamtra.readthedocs.io/en/latest/settings.html

    6. RUN THE SIMULATION
       `pam.runParallelPamtra()` executes the radiative transfer simulation.
       - First argument: array of frequencies in GHz (e.g., [35.5] for MIRA-35).
       - `pp_deltaX`, `pp_deltaY`: Parallelization over spatial dimensions.
         For single-point meteograms, both are 1.
       - `pp_deltaF`: Parallelization over frequencies. Set to 1 for single freq.
       - `pp_local_workers`: Number of CPU cores for parallel processing.

       Reference: https://pamtra.readthedocs.io/en/latest/running.html

    7. SAVE RESULTS
       `pam.writeResultsToNetCDF()` saves all output variables (Ze, MDV, SW, etc.)
       to a NetCDF file.

    Parameters
    ----------
    times_batch : numpy.ndarray
        Timestamps in seconds (Unix time).
    press_batch : numpy.ndarray
        Pressure (Pa), shape (time, levels).
    temp_batch : numpy.ndarray
        Temperature (K), shape (time, levels).
    rh_batch : numpy.ndarray
        Relative humidity (%), shape (time, levels).
    uv_batch : numpy.ndarray
        Horizontal wind speed (m/s), shape (time, levels).
    w_batch : numpy.ndarray
        Vertical wind speed (m/s), shape (time, levels).
    hydro_q : numpy.ndarray
        Hydrometeor mixing ratios (kg/kg), shape (time, levels, n_hydro).
        Order must match the descriptor file.
    hydro_n : numpy.ndarray
        Number concentrations (m^-3), shape (time, levels, n_hydro).
        Required for 2-moment scheme (moment_in = 13).
    zfull_pamtra : numpy.ndarray
        Heights in meters (bottom-up), shape (nlev,).
    lat : float
        Station latitude (degrees).
    lon : float
        Station longitude (degrees).
    descriptorFile : numpy.ndarray
        PAMTRA descriptor file with hydrometeor definitions.
    output_file : str
        Output NetCDF file path.
    cores : int
        Number of CPU cores for parallel processing.
    logger : logging.Logger, optional
        Logger para escrever o acompanhamento em um arquivo de log.

    References
    ----------
    - PAMTRA Documentation: https://pamtra.readthedocs.io
    - PAMTRA Descriptor File: https://pamtra.readthedocs.io/en/latest/descriptor.html
    - PAMTRA Settings: https://pamtra.readthedocs.io/en/latest/settings.html
    - PAMTRA Profile: https://pamtra.readthedocs.io/en/latest/profile.html
    - PAMTRA Running: https://pamtra.readthedocs.io/en/latest/running.html
    - PAMTRA Tutorial: https://atmos.meteo.uni-koeln.de/~mech/pamtra.html
    """
    # ------------------------------------------------------------------
    # 1. PREPARE SPATIAL COORDINATES
    # ------------------------------------------------------------------
    # Height for each time profile: tile the single-column height array
    # to match the number of time steps in the batch.
    # PAMTRA expects heights sorted bottom-up (ascending).
    hgt_batch = np.tile(zfull_pamtra, (len(times_batch), 1))
    lat_batch = np.full(len(times_batch), lat)
    lon_batch = np.full(len(times_batch), lon)

    # ------------------------------------------------------------------
    # 2. CREATE THE PAMTRA OBJECT
    # ------------------------------------------------------------------
    # Initializes an empty PAMTRA instance with default settings.
    pam = pyPamtra.pyPamtra()

    # ------------------------------------------------------------------
    # 3. ADD HYDROMETEORS FROM THE DESCRIPTOR FILE
    # ------------------------------------------------------------------
    # Each row defines one hydrometeor class. The order here must match
    # the order of hydro_q and hydro_n arrays.
    for hydro in descriptorFile:
        pam.df.addHydrometeor(hydro)

    # ------------------------------------------------------------------
    # 4. CREATE THE ATMOSPHERIC PROFILE
    # ------------------------------------------------------------------
    # Builds the atmospheric state from ICON model data.
    # Required: hgt, press, temp, relhum
    # Optional but critical: hydro_q, hydro_n, wind_uv, wind_w
    pam.createProfile(
        hgt=hgt_batch,          # Height [m], bottom-up, shape (ntime, nlev)
        press=press_batch,      # Pressure [Pa], shape (ntime, nlev)
        temp=temp_batch,        # Temperature [K], shape (ntime, nlev)
        relhum=rh_batch,        # Relative humidity [%], shape (ntime, nlev)
        hydro_q=hydro_q,        # Mixing ratios [kg/kg], shape (ntime, nlev, n_hydro)
        hydro_n=hydro_n,        # Number conc [m^-3], shape (ntime, nlev, n_hydro)
        wind_uv=uv_batch,       # Horizontal wind [m/s], affects Doppler broadening
        wind_w=w_batch,         # Vertical wind [m/s], affects Doppler shift
        timestamp=times_batch,  # Unix timestamps [s], for output metadata
        lat=lat_batch,          # Latitude [deg], for output metadata
        lon=lon_batch,          # Longitude [deg], for output metadata
    )

    # ------------------------------------------------------------------
    # 5. CONFIGURE THE MIRA-35 RADAR (35.5 GHz)
    # ------------------------------------------------------------------
    # All radar settings are stored in pam.nmlSet.
    # Reference: https://pamtra.readthedocs.io/en/latest/settings.html

    # Enable active (radar) mode; disable passive (radiometer) mode
    pam.nmlSet["passive"] = False
    pam.nmlSet["active"] = True

    # Radar simulation mode:
    #   'simple'   -> Only Ze (fastest, least detailed)
    #   'spectrum' -> Full Doppler spectrum + all moments (slowest, most memory)
    #   'moments'  -> All moments without saving the full spectrum (balanced)
    pam.nmlSet["radar_mode"] = "moments"

    # MIRA-35 beamwidth: 0.5 degrees (affects spectral broadening)
    pam.nmlSet["radar_fwhr_beamwidth_deg"] = 0.5

    # MIRA-35 integration time: 2.8 seconds (affects velocity resolution)
    pam.nmlSet["radar_integration_time"] = 2.8

    # Nyquist velocity range: ±6.255 m/s per interval
    pam.nmlSet["radar_max_v"] = 6.255
    pam.nmlSet["radar_min_v"] = -6.255

    # Nyquist intervals: 4 -> total unambiguous velocity = 6.255 * 4 ≈ 25 m/s
    # Prevents the error 'increase radar_aliasing_nyquist_interv' when the
    # spectrum velocity exceeds radar_max_v * radar_aliasing_nyquist_interv.
    pam.nmlSet["radar_aliasing_nyquist_interv"] = 4

    # FFT size for spectrum computation: 512 points
    pam.nmlSet["radar_nfft"] = 512

    # Number of spectral averages: 30 (smoother spectrum, longer integration)
    pam.nmlSet["radar_no_ave"] = 30

    # Receiver noise level: -56 dB (MIRA-35 sensitivity)
    pam.nmlSet["radar_pnoise0"] = -56

    # Verbosity: 0 = minimal FORTRAN output, 1 = Python progress
    pam.set['verbose'] = 0
    pam.set['pyVerbose'] = 1

    # Turbulence: constant EDR (Eddy Dissipation Rate) for air motion model
    pam.p['turb_edr'][:] = 1.0e-4

    # Air motion effects on Doppler spectrum: enabled
    pam.nmlSet['radar_airmotion'] = True
    pam.nmlSet['radar_airmotion_model'] = 'constant'

    # Polarization: MIRA-35 is mono-polarized -> 'NN' (non-polarized)
    # For dual-pol: 'HH,VV,HV' (needed for ZDR, LDR, etc.)
    pam.nmlSet['radar_polarisation'] = 'NN'
    #pam.nmlSet['active_polarization'] = 'full'

    # ------------------------------------------------------------------
    # 6. RUN THE SIMULATION
    # ------------------------------------------------------------------
    # Mostra ao usuário qual arquivo está sendo processado
    print(f"  Processando: {os.path.basename(output_file)}")
    
    if logger:
        logger.info("  Running PAMTRA...")
        print_memory_usage("  Before PAMTRA", logger=logger)

    # Execute the radiative transfer simulation at 35.5 GHz.
    # pp_deltaX/Y: spatial parallelization (1 for single-point meteogram)
    # pp_deltaF: frequency parallelization (1 for single frequency)
    # pp_local_workers: number of CPU cores
    pam.runParallelPamtra(
        np.array([35.5]),      # Frequency array in GHz
        pp_deltaX=1,           # Parallelization over X dimension
        pp_deltaY=1,           # Parallelization over Y dimension
        pp_deltaF=1,           # Parallelization over frequencies
        pp_local_workers=cores # Number of CPU cores
    )

    if logger:
        print_memory_usage("  After PAMTRA", logger=logger)

    # ------------------------------------------------------------------
    # 7. SAVE RESULTS
    # ------------------------------------------------------------------
    pam.writeResultsToNetCDF(output_file)
    if logger:
        logger.info(f"  Saved: {output_file}")


def main(input_file=None, microphysics=None, station_index=None,
         output_dir=None, cores=None, processing_flag=None):
    """
    Main function that processes the ICON meteogram and runs PAMTRA.

    Parameters
    ----------
    input_file : str
        Path to the ICON meteogram file.
    microphysics : str
        Microphysics scheme: "1mom", "2mom", or "auto".
    station_index : int
        Station index in the stations array.
    output_dir : str
        Directory for output files.
    cores : int
        Number of cores for parallel processing.
    processing_flag : str
        Processing flag: "raw", "total", "subgrid", "convective".
    """
    # Use initial configuration as default
    if input_file is None:
        input_file = INPUT_FILE
    if microphysics is None:
        microphysics = MICROPHYSICS
    if station_index is None:
        station_index = STATION_INDEX
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if cores is None:
        cores = CORES
    if processing_flag is None:
        processing_flag = PROCESSING_FLAG

    # Validate the processing flag
    valid_flags = ["raw", "total", "subgrid", "convective"]
    if processing_flag not in valid_flags:
        print(f"Error: PROCESSING_FLAG '{processing_flag}' is invalid.", file=sys.stderr)
        print(f"Valid options: {valid_flags}", file=sys.stderr)
        sys.exit(1)

    # Configura o logger para escrever o acompanhamento em um arquivo de log
    logger = setup_logger(output_dir)

    # Mensagem inicial limpa para o usuário
    print("=" * 60)
    print("ICON → PAMTRA: MIRA-35 Radar Simulation (35.5 GHz) — v3")
    print("=" * 60)
    print(f"\nConfiguração:")
    print(f"  Arquivo de entrada: {input_file}")
    print(f"  Período: {START_DATE} até {END_DATE}")
    print(f"  Estação: {station_index}")
    print(f"  Flag de processamento: {processing_flag}")
    print(f"  Microfísica: {microphysics}")
    print(f"  Diretório de saída: {output_dir}")
    print(f"\nLog detalhado: {os.path.join(output_dir, 'pamtra_processing.log')}")

    # Log detalhado no arquivo
    logger.info("=" * 60)
    logger.info("ICON → PAMTRA: MIRA-35 Radar Simulation (35.5 GHz) — v3")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Input file: {input_file}")
    logger.info(f"  Period: {START_DATE} to {END_DATE}")
    logger.info(f"  Station: {station_index}")
    logger.info(f"  Processing flag: {processing_flag}")
    logger.info(f"  Microphysics: {microphysics}")
    logger.info(f"  Output dir: {output_dir}")

    # -------------------------------------------------------
    # 1. Read the meteogram file
    # -------------------------------------------------------
    full_input_path = os.path.join(INPUT_DIR, input_file)
    logger.info(f"\n1. Reading file: {full_input_path}")
    meteog = xr.open_dataset(full_input_path)
    var_names = get_available_variables(meteog)
    logger.info(f"   Available variables: {var_names}")

    # -------------------------------------------------------
    # 2. Filter by period (START_DATE / END_DATE)
    # -------------------------------------------------------
    times_all = get_times(meteog)
    dates_all = pd.to_datetime(times_all, unit="s")

    start_dt = pd.Timestamp(START_DATE)
    end_dt = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    date_mask = (dates_all >= start_dt) & (dates_all <= end_dt)
    logger.info(f"\n2. Filtering period: {START_DATE} to {END_DATE}")
    logger.info(f"   Total time steps: {len(times_all)}")
    logger.info(f"   Filtered time steps: {date_mask.sum()}")

    if date_mask.sum() == 0:
        print("Error: No data found in the specified period!", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------
    # 3. Detect the microphysics scheme
    # -------------------------------------------------------
    if microphysics == "auto":
        if has_2moment_vars(var_names):
            microphysics = "2mom"
            logger.info("\n   Detected: 2-moment scheme (variables NC, NI, etc.)")
        else:
            microphysics = "1mom"
            logger.info("\n   Detected: 1-moment scheme (mixing ratios only)")
    else:
        logger.info(f"\n   Selected scheme: {microphysics}-moment")

    # -------------------------------------------------------
    # 4. Select the descriptor file
    # -------------------------------------------------------
    # The descriptor file is determined solely by the microphysics scheme.
    # 1-moment -> descriptorFile_gscp2 (5 hydrometeors, moment_in = 3)
    # 2-moment -> descriptorFile_gscp4 (6 hydrometeors + hail, moment_in = 13)
    if microphysics == "2mom":
        descriptorFile = get_descriptor_2moment()
        n_hydro = 6  # cloud, ice, rain, snow, graupel, hail
        logger.info(f"\n4. Descriptor file: 2-moment (gscp4, {n_hydro} hydrometeors incl. hail)")
    else:
        descriptorFile = get_descriptor_1moment()
        n_hydro = 5  # cloud, ice, rain, snow, graupel
        logger.info(f"\n4. Descriptor file: 1-moment (gscp2, {n_hydro} hydrometeors)")

    # -------------------------------------------------------
    # 5. Extract variables
    # -------------------------------------------------------
    logger.info(f"\n5. Extracting variables for station {station_index}")
    vars_meteog = get_meteog_variables(meteog, station_index)

    # Find the number of levels
    for i in range(meteog.sizes["nvars"]):
        name = meteog["var_name"].values[i].decode()
        if name == "RHO":
            nlev = int(meteog["var_nlevs"].values[i])
            break

    logger.info(f"   Number of levels (full levels): {nlev}")

    # Extract full level variables
    press = vars_meteog["P"][:, :nlev].values
    temp = vars_meteog["T"][:, :nlev].values
    rho = vars_meteog["RHO"][:, :nlev].values
    rh = vars_meteog["REL_HUM"][:, :nlev].values
    qv = vars_meteog["QV"][:, :nlev].values

    # Hydrometeors - copy for modification
    qc_raw = vars_meteog["QC"][:, :nlev].values.copy()
    qi_raw = vars_meteog["QI"][:, :nlev].values.copy()
    qr_raw = vars_meteog["QR"][:, :nlev].values.copy()
    qs_raw = vars_meteog["QS"][:, :nlev].values.copy()
    qg_raw = vars_meteog["QG"][:, :nlev].values.copy()

    u = vars_meteog["U"][:, :nlev].values
    v = vars_meteog["V"][:, :nlev].values

    # Heights in full levels
    #
    # CRITICAL: ICON stores vertical profiles TOP-to-BOTTOM.
    # The first index (0) is the model top (upper atmosphere),
    # and the last index (nlev-1) is the surface.
    #
    # PAMTRA expects profiles BOTTOM-to-TOP (surface first, ascending height).
    # Therefore, we MUST reverse ALL vertical arrays before passing them to PAMTRA.
    # If only the heights are reversed but not the data variables, PAMTRA will
    # associate the wrong temperature/hydrometeor values with each altitude,
    # producing physically incorrect radar simulations.
    #
    # The reversal is applied below to: zfull, press, temp, rh, qv, qc, qi, qr,
    # qs, qg, qh, u, v, w, and all hydro_q/hydro_n arrays.
    zfull = meteog["heights"][:nlev, 0, 0].values
    # Reverse from ICON order (top->bottom) to PAMTRA order (bottom->top)
    zfull_pamtra = zfull[::-1]

    # ========================================================================
    # REVERSE ALL FULL-LEVEL VARIABLES: ICON (top->bottom) -> PAMTRA (bottom->top)
    # ========================================================================
    # Because ICON stores the vertical dimension with index 0 at the model top
    # and index nlev-1 at the surface, every full-level variable must be
    # reversed along the vertical axis (axis=1) before PAMTRA receives it.
    # The height array (zfull_pamtra) is already reversed above.
    #
    # If this step is skipped, PAMTRA will associate the surface pressure
    # with the upper-atmosphere temperature, the cloud-top hydrometeors with
    # the ground, etc. — producing completely wrong radar simulations.
    # ========================================================================

    press = press[:, ::-1]
    temp = temp[:, ::-1]
    rho = rho[:, ::-1]
    rh = rh[:, ::-1]
    qv = qv[:, ::-1]

    qc_raw = qc_raw[:, ::-1]
    qi_raw = qi_raw[:, ::-1]
    qr_raw = qr_raw[:, ::-1]
    qs_raw = qs_raw[:, ::-1]
    qg_raw = qg_raw[:, ::-1]

    u = u[:, ::-1]
    v = v[:, ::-1]

    # Half level variables — also reversed (rain_con, snow_con, w)
    # Note: half levels have shape (time, nlev+1). Reverse the vertical axis.
    rain_con = vars_meteog["RAIN_CON_RATE_3D"][:, :].values[:, ::-1]
    snow_con = vars_meteog["SNOW_CON_RATE_3D"][:, :].values[:, ::-1]
    w = vars_meteog["W"][:, :].values[:, ::-1]

    # Diagnostic variables (subgrid)
    qc_diag = vars_meteog["QC_DIA"][:, :nlev].values[:, ::-1]
    qi_diag = vars_meteog["QI_DIA"][:, :nlev].values[:, ::-1]

    # Coordinates and time
    lon = float(meteog["station_lon"][station_index].values)
    lat = float(meteog["station_lat"][station_index].values)

    station_name = meteog["station_name"].values[station_index].decode().strip()
    logger.info(f"   Station: {station_name} (lat={lat:.2f}, lon={lon:.2f})")

    # -------------------------------------------------------
    # 6. Pre-processing based on FLAG
    # -------------------------------------------------------
    logger.info(f"\n6. Pre-processing (flag: {processing_flag})...")

    # Horizontal wind speed
    uv = np.hypot(u, v)

    # Interpolate W from half levels to full levels
    w_full = 0.5 * (w[:, :-1] + w[:, 1:])

    # Convective contributions (always computed, applied according to flag)
    qrcon, qscon, nrcon, nscon = process_convective_precipitation(
        rain_con, snow_con, temp, rho, True
    )

    # Subgrid contributions
    qc_sgs = qc_diag  # max(0, qc_diag - qc)
    qi_sgs = qi_diag - qi_raw - qs_raw          # qi_diag - qi - qs

    # Small epsilon to avoid divide-by-zero warnings in nisgs calculation
    eps = 1e-15

    # Initialize working variables
    qc = qc_raw.copy()
    qi = qi_raw.copy()
    qr = qr_raw.copy()
    qs = qs_raw.copy()
    qg = qg_raw.copy()

    nc = np.zeros_like(qc)
    nr = np.zeros_like(qr)
    ni = np.zeros_like(qi)
    ns = np.zeros_like(qs)
    ng = np.zeros_like(qg)

    # For 2-moment scheme, also initialize hail variables
    qh = np.zeros_like(qc)
    nh = np.zeros_like(qc)

    if processing_flag == "raw":
        # (a) Raw ICON data, no corrections
        logger.info("   FLAG_RAW: Using raw ICON data")
        qc = qc_raw.copy()
        qi = qi_raw.copy()
        qr = qr_raw.copy()
        qs = qs_raw.copy()
        qg = qg_raw.copy()

        if microphysics == "2mom":
            # 2-moment: uses model concentrations (reversed top->bottom to bottom->top)
            if "QH" in var_names:
                qh = vars_meteog["QH"][:, :nlev].values[:, ::-1].copy()
            if "NC" in var_names:
                nc = vars_meteog["NC"][:, :nlev].values[:, ::-1].copy()
            if "NR" in var_names:
                nr = vars_meteog["NR"][:, :nlev].values[:, ::-1].copy()
            if "NI" in var_names:
                ni = vars_meteog["NI"][:, :nlev].values[:, ::-1].copy()
            if "NS" in var_names:
                ns = vars_meteog["NS"][:, :nlev].values[:, ::-1].copy()
            if "NG" in var_names:
                ng = vars_meteog["NG"][:, :nlev].values[:, ::-1].copy()
            if "NH" in var_names:
                nh = vars_meteog["NH"][:, :nlev].values[:, ::-1].copy()
            logger.info("   Model number concentrations (2-moment, incl. hail)")

    elif processing_flag == "total":
        # (b) Subgrid + convective effects
        logger.info("   FLAG_TOTAL: Subgrid + convective")

        # Mixing ratios
        qc = qc_sgs
        qr = qr_raw + qrcon
        qi = qi_raw + qi_sgs
        qs = qs_raw + qscon

        if microphysics == "1mom":
            # 1-moment: zero number concentrations (PSD from descriptor file)
            nc = np.zeros_like(qc)
            ni = np.zeros_like(qi)
            nr = np.zeros_like(qr)
            ns = np.zeros_like(qs)
            ng = np.zeros_like(qg)
            logger.info("   Zero number concentrations (1-moment, PSD from descriptor)")
        else:
            # 2-moment: uses model concentrations + convective (reversed top->bottom to bottom->top)
            if "NC" in var_names:
                nc = vars_meteog["NC"][:, :nlev].values[:, ::-1].copy()
            if "NR" in var_names:
                nr = vars_meteog["NR"][:, :nlev].values[:, ::-1].copy()
            else:
                nr = np.zeros_like(qr)
            if "NI" in var_names:
                ni = vars_meteog["NI"][:, :nlev].values[:, ::-1].copy()
            else:
                ni = np.zeros_like(qi)
            if "NS" in var_names:
                ns = vars_meteog["NS"][:, :nlev].values[:, ::-1].copy()
            else:
                ns = np.zeros_like(qs)
            if "NG" in var_names:
                ng = vars_meteog["NG"][:, :nlev].values[:, ::-1].copy()
            else:
                ng = np.zeros_like(qg)
            # Hail: extract from model if available (reversed)
            if "QH" in var_names:
                qh = vars_meteog["QH"][:, :nlev].values[:, ::-1].copy()
            if "NH" in var_names:
                nh = vars_meteog["NH"][:, :nlev].values[:, ::-1].copy()

            # Add subgrid ice number concentration (nisgs) following
            # pamtra_earthcare_track.py pattern for 2-moment schemes.
            # This computes the number concentration of subgrid-scale ice
            # based on whether grid-scale ice or subgrid ice dominates.
            nisgs = np.zeros_like(qi_sgs)
            qigrd = np.where((qi > eps), qi, np.nan)  # avoid divide by zero warning
            nisgs = np.where(
                (qi > qi_sgs) & (qi > eps),
                ni * qi_sgs / qigrd,
                nisgs
            )  # use size information of grid-scale ice if grid-scale ice dominates
            nisgs = np.where(
                (qi_sgs > qi) & (qi_sgs > 0.0),
                nicooper(temp),
                nisgs
            )  # use Cooper curve for sgs ice only if sgs ice dominates
            ni = ni + nisgs

            # Add convective contributions to concentrations
            nr = nr + nrcon
            ns = ns + nscon
            logger.info("   Model number concentrations + convective + subgrid ice (2-moment, incl. hail)")

    elif processing_flag == "subgrid":
        # (c) Subgrid-scale contributions only
        logger.info("   FLAG_SUBGRID: Subgrid-scale only")

        qc = qc_sgs
        qi = qi_raw + qi_sgs
        qr = qr_raw.copy()
        qs = qs_raw.copy()

        if microphysics == "1mom":
            # 1-moment: zero number concentrations (PSD from descriptor file)
            nc = np.zeros_like(qc)
            ni = np.zeros_like(qi)
            nr = np.zeros_like(qr)
            ns = np.zeros_like(qs)
            ng = np.zeros_like(qg)
            logger.info("   Zero number concentrations (1-moment, PSD from descriptor)")
        else:
            # 2-moment: uses model concentrations (reversed top->bottom to bottom->top)
            if "NC" in var_names:
                nc = vars_meteog["NC"][:, :nlev].values[:, ::-1].copy()
            if "NR" in var_names:
                nr = vars_meteog["NR"][:, :nlev].values[:, ::-1].copy()
            if "NI" in var_names:
                ni = vars_meteog["NI"][:, :nlev].values[:, ::-1].copy()
            if "NS" in var_names:
                ns = vars_meteog["NS"][:, :nlev].values[:, ::-1].copy()
            if "NG" in var_names:
                ng = vars_meteog["NG"][:, :nlev].values[:, ::-1].copy()
            # Hail: extract from model if available (reversed)
            if "QH" in var_names:
                qh = vars_meteog["QH"][:, :nlev].values[:, ::-1].copy()
            if "NH" in var_names:
                nh = vars_meteog["NH"][:, :nlev].values[:, ::-1].copy()

            # Add subgrid ice number concentration (nisgs) following
            # pamtra_earthcare_track.py pattern for 2-moment schemes.
            # For the subgrid flag, qi and qi_sgs represent only the subgrid
            # contribution, so the nisgs calculation adapts accordingly.
            nisgs = np.zeros_like(qi_sgs)
            qigrd = np.where((qi > eps), qi, np.nan)  # avoid divide by zero warning
            nisgs = np.where(
                (qi > qi_sgs) & (qi > eps),
                ni * qi_sgs / qigrd,
                nisgs
            )  # use size information of grid-scale ice if grid-scale ice dominates
            nisgs = np.where(
                (qi_sgs > qi) & (qi_sgs > 0.0),
                nicooper(temp),
                nisgs
            )  # use Cooper curve for sgs ice only if sgs ice dominates
            ni = ni + nisgs
            logger.info("   Model number concentrations + subgrid ice (2-moment, incl. hail)")

    elif processing_flag == "convective":
        # (d) Convective effects only
        logger.info("   FLAG_CONVECTIVE: Convective effects only")

        qc = qc_raw.copy()
        qi = qi_raw.copy()
        qr = qr_raw + qrcon
        qs = qs_raw + qscon

        if microphysics == "1mom":
            # 1-moment: zero number concentrations (PSD from descriptor file)
            nc = np.zeros_like(qc)
            ni = np.zeros_like(qi)
            nr = np.zeros_like(qr)
            ns = np.zeros_like(qs)
            ng = np.zeros_like(qg)
            logger.info("   Zero number concentrations (1-moment, PSD from descriptor)")
        else:
            # 2-moment: uses model concentrations + convective (reversed top->bottom to bottom->top)
            if "NC" in var_names:
                nc = vars_meteog["NC"][:, :nlev].values[:, ::-1].copy()
            if "NR" in var_names:
                nr = vars_meteog["NR"][:, :nlev].values[:, ::-1].copy() + nrcon
            if "NI" in var_names:
                ni = vars_meteog["NI"][:, :nlev].values[:, ::-1].copy()
            if "NS" in var_names:
                ns = vars_meteog["NS"][:, :nlev].values[:, ::-1].copy() + nscon
            if "NG" in var_names:
                ng = vars_meteog["NG"][:, :nlev].values[:, ::-1].copy()
            # Hail: extract from model if available (reversed)
            if "QH" in var_names:
                qh = vars_meteog["QH"][:, :nlev].values[:, ::-1].copy()
            if "NH" in var_names:
                nh = vars_meteog["NH"][:, :nlev].values[:, ::-1].copy()
            logger.info("   Model number concentrations + convective (2-moment, incl. hail)")

    # -------------------------------------------------------
    # 7. Assemble arrays for PAMTRA
    # -------------------------------------------------------
    logger.info("\n7. Assembling arrays for PAMTRA...")

    if microphysics == "2mom":
        # 2-moment: 6 hydrometeors (cloud, ice, rain, snow, graupel, hail)
        hydro_q_full = np.stack([qc, qi, qr, qs, qg, qh], axis=-1)
        hydro_n_full = np.stack([nc, ni, nr, ns, ng, nh], axis=-1)
    else:
        # 1-moment: 5 hydrometeors (cloud, ice, rain, snow, graupel)
        hydro_q_full = np.stack([qc, qi, qr, qs, qg], axis=-1)
        hydro_n_full = np.stack([nc, ni, nr, ns, ng], axis=-1)

    # Ensure mixing ratios are non-negative
    hydro_q_full[hydro_q_full < 0.0] = 0.0

    # PAMTRA upper limit for mixing ratio
    hydro_q_full[hydro_q_full > 0.03] = 0.03

    # Replace invalid values in number concentrations
    hydro_n_full = np.nan_to_num(
        hydro_n_full, nan=1e3, posinf=1e3, neginf=1e3
    )

    # Check for NaNs
    if np.isnan(hydro_q_full).any():
        print("Error: NaN detected in hydro_q_full!", file=sys.stderr)
        sys.exit(1)

    if np.isnan(hydro_n_full).any():
        print("Error: NaN detected in hydro_n_full!", file=sys.stderr)
        sys.exit(1)

    logger.info(f"   hydro_q shape: {hydro_q_full.shape}")
    logger.info(f"   hydro_n shape: {hydro_n_full.shape}")

    # Sanity check
    logger.info("\n   Sanity check:")
    logger.info(f"     max press = {np.max(press):.3e} Pa")
    logger.info(f"     max temp  = {np.max(temp):.3e} K")
    logger.info(f"     max qv    = {np.max(qv):.3e} kg/kg")
    logger.info(f"     max qc    = {np.max(qc):.3e} kg/kg")
    logger.info(f"     max qr    = {np.max(qr):.3e} kg/kg")
    logger.info(f"     max qi    = {np.max(qi):.3e} kg/kg")
    logger.info(f"     max qs    = {np.max(qs):.3e} kg/kg")
    logger.info(f"     max qg    = {np.max(qg):.3e} kg/kg")
    if microphysics == "2mom":
        logger.info(f"     max qh    = {np.max(qh):.3e} kg/kg")

    # -------------------------------------------------------
    # 8. Filter by date and run PAMTRA day by day
    # -------------------------------------------------------
    logger.info("\n8. Running PAMTRA day by day...")

    # Apply the date mask
    times = times_all[date_mask]
    press_filtered = press[date_mask, :]
    temp_filtered = temp[date_mask, :]
    rh_filtered = rh[date_mask, :]
    uv_filtered = uv[date_mask, :]
    w_filtered = w_full[date_mask, :]
    hydro_q_filtered = hydro_q_full[date_mask, :, :]
    hydro_n_filtered = hydro_n_full[date_mask, :, :]

    dates = pd.to_datetime(times, unit="s")
    days = dates.normalize()
    unique_days = days.unique()

    logger.info(f"   Total days: {len(unique_days)}")

    for day_idx, day in enumerate(unique_days):
        logger.info(f"\n   Day {day_idx + 1}/{len(unique_days)}: "
              f"{day.strftime('%Y-%m-%d')}")

        # Mask for the current day
        day_mask = (days == day)

        # Extract the day's data
        times_batch = times[day_mask]
        press_batch = press_filtered[day_mask, :]
        temp_batch = temp_filtered[day_mask, :]
        rh_batch = rh_filtered[day_mask, :]
        uv_batch = uv_filtered[day_mask, :]
        w_batch = w_filtered[day_mask, :]
        hydro_q = hydro_q_filtered[day_mask, :, :]
        hydro_n = hydro_n_filtered[day_mask, :, :]

        # Output file name
        output_file = os.path.join(
            output_dir,
            f"MIRA35_{station_name}_{day.strftime('%Y%m%d')}.nc"
        )

        # Run PAMTRA
        run_pamtra_daily(
            times_batch, press_batch, temp_batch, rh_batch,
            uv_batch, w_batch, hydro_q, hydro_n,
            zfull_pamtra, lat, lon, descriptorFile,
            output_file, cores=cores, logger=logger
        )

    logger.info("\n" + "=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Files saved in: {output_dir}")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("Processamento concluído!")
    print(f"Arquivos salvos em: {output_dir}")
    print(f"Log detalhado: {os.path.join(output_dir, 'pamtra_processing.log')}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts ICON meteogram to MIRA-35 radar simulation via PAMTRA (v3)"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help=f"Path to the ICON meteogram file (.nc) (default: {INPUT_FILE})"
    )
    parser.add_argument(
        "--microphysics", "-m",
        choices=["1mom", "2mom", "auto"],
        default=None,
        help=f"Microphysics scheme (default: {MICROPHYSICS})"
    )
    parser.add_argument(
        "--station", "-s",
        type=int,
        default=None,
        help=f"Station index (default: {STATION_INDEX})"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help=f"Directory for output files (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--cores", "-c",
        type=int,
        default=None,
        help=f"Number of cores for parallel processing (default: {CORES})"
    )
    parser.add_argument(
        "--flag", "-f",
        choices=["raw", "total", "subgrid", "convective"],
        default=None,
        help=f"Pre-processing flag (default: {PROCESSING_FLAG})"
    )

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        microphysics=args.microphysics,
        station_index=args.station,
        output_dir=args.output_dir,
        cores=args.cores,
        processing_flag=args.flag
    )
