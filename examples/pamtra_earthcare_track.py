import os
import sys
os.environ['PAMTRA_DATADIR'] = '' # workaround 
import pyPamtra
import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
from types import SimpleNamespace
import psutil

# Get current process
process = psutil.Process(os.getpid())

print("This is Pamtra for EarthCARE")

print("Pamtra shared object file is")
print(pyPamtra.pyPamtraLib.__file__)

if len(sys.argv) == 5:
    datadir    = sys.argv[1]  
    igscp      = int(sys.argv[2])
    datestring = sys.argv[3]  
    nout_start = int(sys.argv[4])   # only one step
    nout_end   = nout_start
    print(f"  {nout_start=}")
elif len(sys.argv) == 6:
    datadir    = sys.argv[1]  
    igscp      = int(sys.argv[2])
    datestring = sys.argv[3]  
    nout_start = int(sys.argv[4])   
    nout_end   = int(sys.argv[5])   
    print(f"  Range {nout_start=} to {nout_end=}")
elif len(sys.argv) == 4:
    datadir    = sys.argv[1]  
    igscp      = int(sys.argv[2])
    datestring = sys.argv[3]  
    print(f"  Loop over all columns steps in dataset")
    nout_start = -1
    nout_end   = -2
else:
    print("Usage: python pamtra_st.py datadir date chunk")
    sys.exit(1)

print("   date = "+datestring)

# 2mom scheme particle definitions directly copied from ICON code
cloud = SimpleNamespace(
    name="cloud_nue1mue1",  # name: name/identifier of the particle class
    nu=1.000000,            # nu: width parameter of the size distribution
    mu=1.000000,            # mu: exponent parameter of the size distribution
    x_max=2.60e-10,         # x_max: maximum particle mass (D = 80 µm)
    x_min=4.20e-15,         # x_min: minimum particle mass (D = 2 µm)
    a_geo=1.24e-1,          # a_geo: geometry coefficient
    b_geo=0.333333,         # b_geo: geometry exponent (= 1/3)
    a_ms=-1.0,              # a_ms: m-D-relation for pamtra
    b_ms=-1.0,              # b_ms: m-D-relation for pamtra
    a_vel=3.75e5,           # a_vel: coefficient for fall-speed relation
    b_vel=0.666667,         # b_vel: exponent for fall-speed relation
    a_ven=0.780000,         # a_ven: coefficient for ventilation (PK, p.541)
    b_ven=0.308000,         # b_ven: exponent for ventilation (PK, p.541)
    cap=2.00,               # cap: capacitance coefficient
    vsedi_max=1.0,          # vsedi_max: maximum sedimentation velocity
    vsedi_min=0.0           # vsedi_min: minimum sedimentation velocity
)
rain = SimpleNamespace(
    name="rainSBB",        # name: name/identifier of the particle class
    nu=1.000000,           # nu: width parameter of the size distribution
    mu=0.333333,           # mu: exponent parameter of the size distribution
    x_max=6.50e-5,         # x_max: maximum particle mass (~5 mm diameter)
    x_min=2.60e-10,        # x_min: minimum particle mass (~80 µm diameter)
    a_geo=1.24e-1,         # a_geo: geometry coefficient
    b_geo=0.333333,        # b_geo: geometry exponent (= 1/3)
    a_ms=-1.0,             # a_ms: m-D-relation for pamtra
    b_ms=-1.0,             # b_ms: m-D-relation for pamtra
    a_vel=114.0137,        # a_vel: coefficient for fall-speed relation
    b_vel=0.234370,        # b_vel: exponent for fall-speed relation
    a_ven=0.780000,        # a_ven: coefficient for ventilation (PK, p.541)
    b_ven=0.308000,        # b_ven: exponent for ventilation (PK, p.541)
    cap=2.000000,          # cap: capacitance coefficient
    vsedi_max=20.0,        # vsedi_max: maximum sedimentation velocity
    vsedi_min=0.1          # vsedi_min: minimum sedimentation velocity
)
ice = SimpleNamespace(
    name="ice_cosmo5",     # name: name/identifier of the particle class
    nu=0.000000,           # nu: width parameter of the size distribution
    mu=0.333333,           # mu: exponent parameter of the size distribution
    x_max=1.00e-5,         # x_max: maximum particle mass (approx. diameter unknown)
    x_min=1.00e-12,        # x_min: minimum particle mass (~200 µm diameter)
    a_geo=0.835000,        # a_geo: geometry coefficient
    b_geo=0.390000,        # b_geo: geometry exponent
    a_ms=-1.0,             # a_ms: m-D-relation for pamtra
    b_ms=-1.0,             # b_ms: m-D-relation for pamtra
    a_vel=27.7,            # a_vel: coefficient for fall-speed relation
    b_vel=0.215790,        # b_vel: exponent for fall-speed relation (= 0.41/1.9)
    a_ven=0.780000,        # a_ven: coefficient for ventilation (PK, p.541)
    b_ven=0.308000,        # b_ven: exponent for ventilation (PK, p.541)
    cap=3.0,               # cap: capacitance coefficient
    vsedi_max=3.0,         # vsedi_max: maximum sedimentation velocity
    vsedi_min=0.0,         # vsedi_min: minimum sedimentation velocity
    ecoll_c=0.80,          # ecoll_c: collection efficiency coefficient
    D_crit_c=150.0e-6,     # D_crit_c: critical particle diameter for collection
    q_crit_c=1.000e-5,     # q_crit_c: critical mass mixing ratio for collection
    sigma_vel=0.25         # sigma_vel: velocity variance (spread of fall speeds)
)
snow = SimpleNamespace(
    name="snowSBB",        # name: name/identifier of the particle class
    nu=0.000000,           # nu: width parameter of the size distribution
    mu=0.500000,           # mu: exponent parameter of the size distribution
    x_max=2.00e-5,         # x_max: maximum particle mass
    x_min=1.00e-10,        # x_min: minimum particle mass
    a_geo=5.130000,        # a_geo: geometry coefficient
    b_geo=0.500000,        # b_geo: geometry exponent
    a_ms=-1.0,             # a_ms: m-D-relation for pamtra
    b_ms=-1.0,             # b_ms: m-D-relation for pamtra
    a_vel=400.0000,        # a_vel: coefficient for fall-speed relation 
    b_vel=0.350000,        # b_vel: exponent for fall-speed relation 
    a_ven=0.780000,        # a_ven: coefficient for ventilation (PK, p.541)
    b_ven=0.308000,        # b_ven: exponent for ventilation (PK, p.541)
    cap=3.00,              # cap: capacitance coefficient
    vsedi_max=3.0,         # vsedi_max: maximum sedimentation velocity
    vsedi_min=0.1,         # vsedi_min: minimum sedimentation velocity
    ecoll_c=0.80,          # ecoll_c: collection efficiency coefficient
    D_crit_c=150.0e-6,     # D_crit_c: critical particle diameter for collection
    q_crit_c=1.000e-5,     # q_crit_c: critical mass mixing ratio for collection
    sigma_vel=0.25         # sigma_vel: velocity variance (spread of fall speed)
)
graupel = SimpleNamespace(
    name="graupelhail_cosmo5",  # name: identifier of the particle class
    nu=1.000000,                # nu: 1st shape parameter of the size distribution
    mu=0.333333,                # mu: 2nd shape parameter of the size distribution
    x_max=5.30e-4,              # x_max: maximum particle mean mass
    x_min=4.19e-9,              # x_min: minimum particle mean mass
    a_geo=1.42e-1,              # a_geo: particle geometry prefactor
    b_geo=0.314000,             # b_geo: particle geometry exponent (= 1/3.10)
    a_ms=-1.0,             # a_ms: m-D-relation for pamtra
    b_ms=-1.0,             # b_ms: m-D-relation for pamtra
    a_vel=100.0,                # a_vel: terminal fall velocity prefactor
    b_vel=0.34,                 # b_vel: terminal fall velocity exponent
    a_ven=0.780000,             # a_ven: 1st ventilation coefficient (PK, p.541)
    b_ven=0.308000,             # b_ven: 2nd ventilation coefficient (PK, p.541)
    cap=2.00,                   # cap: capacity coefficient
    vsedi_max=80.0,             # vsedi_max: maximum bulk sedimentation velocity
    vsedi_min=0.10,             # vsedi_min: minimum bulk sedimentation velocity
    ecoll_c=1.0,                # ecoll_c: maximum collision efficiency with cloud droplets
    D_crit_c=100.0e-6,          # D_crit_c: diameter threshold for cloud riming
    q_crit_c=1.000e-6,          # q_crit_c: mixing-ratio threshold for cloud riming
    sigma_vel=0.0               # sigma_vel: dispersion of fall velocity (for collection kernel)
)
hail = SimpleNamespace(
    name="hail_cosmo5",     # name: identifier of the particle class
    nu=1.000000,            # nu: width parameter of the size distribution
    mu=0.333333,            # mu: exponent parameter of the size distribution
    x_max=5.00e-3,          # x_max: maximum particle mass
    x_min=2.60e-9,          # x_min: minimum particle mass
    a_geo=0.1366,           # a_geo: particle geometry prefactor
    b_geo=0.333333,         # b_geo: particle geometry exponent (= 1/3)
    a_ms=-1.0,              # a_ms: m-D-relation for pamtra
    b_ms=-1.0,              # b_ms: m-D-relation for pamtra
    a_vel=39.3,             # a_vel: coefficient for fall-speed relation
    b_vel=0.166667,         # b_vel: exponent for fall-speed relation
    a_ven=0.780000,         # a_ven: 1st ventilation coefficient (PK, p.541)
    b_ven=0.308000,         # b_ven: 2nd ventilation coefficient (PK, p.541)
    cap=2.00,               # cap: capacitance coefficient
    vsedi_max=30.0,         # vsedi_max: maximum sedimentation velocity
    vsedi_min=0.1,          # vsedi_min: minimum sedimentation velocity
    ecoll_c=1.0,            # ecoll_c: maximum collision efficiency with cloud droplets
    D_crit_c=100.0e-6,      # D_crit_c: diameter threshold for cloud riming
    q_crit_c=1.000e-6,      # q_crit_c: mixing-ratio threshold for cloud riming
    sigma_vel=0.0           # sigma_vel: dispersion of fall velocity (for collection kernel)
)

def set_pamtra_geo(particle):
    """
    Convert 2-moment scheme geometry (D = a_geo * m**b_geo) to PAMTRA mass-size (m = a_ms * D**b_ms)
    and 2mom f(x)=A*x**nu*exp(-B*x**mu) to f(D)=N_0*D**nu*exp(-lambda*D**mu)
    """
    # the other geometry exponents
    particle.b_ms = 1.0 / particle.b_geo
    particle.a_ms = particle.a_geo**(-1.0/particle.b_geo)
    # from f(x) to f(D), here we just re-use the variables
    particle.nu=particle.b_ms*(particle.nu+1.0)-1.0
    particle.mu=particle.mu*particle.b_ms
    return

def limit_number_density(q, n, x_min, x_max):
    """
    Limit particle mean mass to [x_min, x_max] by adjusting number density n.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute diameters safely, avoid divide by zero
        x = np.where( (n > 1.0) & (q > 1e-12), q / n, 0.0)  
        # Clip diameters strictly to limits
        x = np.clip(x, x_min, x_max)
        # Recalculate number density where diameter > 0
        n_limited = np.where(x > 0.0, q / x, 0.0)

    return n_limited

def nicooper(temp):
    ni = 5.0E+0 * np.exp(0.304 * (273.15 - temp))   # Cooper (1986)
    return ni

def clean_dataset(ds):
    # this is necessary because NCL write some HDF5 attributes in a way that confuses xarray 
    ds = ds.copy(deep=True)
    for var_name in ds.data_vars:
        for attr in list(ds[var_name].attrs):
            if attr.startswith("_Netcdf4") or attr.startswith("_FillValue_original"):
                del ds[var_name].attrs[attr]
    for coord in ds.coords:
         for attr in ["_Netcdf4Dimid", "CLASS", "NAME", "_NetCDF4Dimid"]:
           if attr in ds[coord].attrs:
             del ds[coord].attrs[attr]
    # Drop both the redundant coordinate and the associated variable
    # (we chunk with 'time' so those end up being wrong anyway)
    # for var in ['time_icon', 'time_idx']:
    #     if var in ds:
    #         ds = ds.drop_vars(var)

    return ds

def clean_for_netcdf(ds: xr.Dataset, verbose=True) -> xr.Dataset:
    """
    Clean an xarray.Dataset to avoid NetCDF write conflicts.

    Fixes:
      - Attribute names that conflict with variable or dimension names
      - Invalid attribute types (lists, dicts, sets)
      - Renames variables that conflict with dimensions (if not coordinate variables)
    """

    ds = ds.copy(deep=True)
    var_names = set(ds.variables)
    dim_names = set(ds.dims)
    global_attrs = set(ds.attrs)

    # 1. Remove problematic global attributes
    for attr in list(ds.attrs):
        if attr in var_names | dim_names:
            if verbose:
                print(f"Removing global attribute '{attr}' (conflicts with variable/dimension name)")
            del ds.attrs[attr]
        elif isinstance(ds.attrs[attr], (list, dict, set)):
            if verbose:
                print(f"Removing non-serializable global attribute '{attr}' ({type(ds.attrs[attr])})")
            del ds.attrs[attr]

    # 2. Remove problematic variable attributes
    for var_name, var in ds.variables.items():
        for attr in list(var.attrs):
            if attr in var_names | dim_names:
                if verbose:
                    print(f"Removing attribute '{attr}' from variable '{var_name}' (name conflict)")
                del var.attrs[attr]
            elif isinstance(var.attrs[attr], (list, dict, set)):
                if verbose:
                    print(f"Removing non-serializable attribute '{attr}' from '{var_name}' ({type(var.attrs[attr])})")
                del var.attrs[attr]

    # 3. Rename variables that conflict with dimensions
    rename_map = {}
    for name in var_names & dim_names:
        # Only rename if not a proper coordinate variable
        if not (name in ds.coords and list(ds[name].dims) == [name]):
            new_name = f"{name}_var"
            rename_map[name] = new_name
            if verbose:
                print(f"Renaming variable '{name}' -> '{new_name}' (dimension conflict)")

    if rename_map:
        ds = ds.rename_vars(rename_map)

    if verbose:
        print("Dataset cleaned and ready for .to_netcdf()")

    return ds

def print_memory_usage(label=""):
    mem_info = process.memory_info()
    print(f"{label} RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB")

    
# Define descriptorFiles
descriptorFile_gscp3 = np.array([ 
   (  # cloud liquid water using 1mom mass
      'cwc_q', 1, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1,
      'mono', -99.0, -99.0, -99.0, -99.0, 2.0e-5, -99.0,
      'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0
   ),
   (  # cloud ice using 2mom with number and mass concentrations (as indicated by 13 in column 9)
      'iwc_q', 1, -1, -99.0, 130.0, 3.0, 0.684, 2.0, 13, 100,
      'mgamma', -99.0, -99.0, 6.0, 6.0, 2.0e-6, 250e-6,
      'ssrg-rt3_0.18_0.89_2.06_0.08', 'corPowerLaw_30.606_0.5533', -99.0
   ),
   (  # rain using 1mom mass
      'rwc_q', 1, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 100,
      'exp', -99.0, -99.0, 8.0e6, -99.0, 1.2e-4, 6.0e-3,
      'mie-sphere', 'corPowerLaw_130.0_0.5', -99.0
   ),
   (  # snow using 1mom mass
      'swc_q', 1, -1, -99.0, 0.038, 2.0, 0.3971, 1.88, 3, 100,
      'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, 5.1e-11, 1.0e-2,
      'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0
   ),
   (  # graupel with N0g=4e6, rho=400, b=3.0
      'gwc_q', 1, -1,  -99.0,  209.4,  3.0,  0.5,  10.,  3, 100,
      'exp',  -99.0,  -99.0,  4.0e6,  -99.0,  8.0e-11,  5.0e-3,
      'mie-sphere', 'corPowerLaw_5.0_0.20', -99.0
   ),
])

descriptorFile_gscp1 = np.array([ 
   (  # cloud liquid water using 1mom mass
      'cwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 1,
      'mono', -99.0, -99.0, -99.0, -99.0, 2.0e-5, -99.0,
      'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0
   ),
   (  # cloud ice using 1mom mass
     'iwc_q', 0.2, -1, -99.0,   130.0,   3.0,  0.684,   2.0,  3,  1,   
     'mono_cosmo_ice', -99.0, -99.0, -99.0, -99.0,   -99.0,  -99.0, 
     'ssrg-rt3_0.18_0.89_2.06_0.08', 'corPowerLaw_30.606_0.5533', -99.0
   ),
   (  # rain using 1mom mass
      'rwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 3, 100,
      'exp', -99.0, -99.0, 8.0e6, -99.0, 1.2e-4, 6.0e-3,
      'mie-sphere', 'corPowerLaw_130.0_0.5', -99.0
   ),
   (  # snow using 1mom mass
      'swc_q', 0.6, -1, -99.0, 0.038, 2.0, 0.3971, 1.88, 3, 100,
      'exp_cosmo_snow', -99.0, -99.0, -99.0, -99.0, 5.1e-11, 1.0e-2,
      'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0
   ),
   (  # graupel with N0g=4e6, rho=400, b=3.0
      'gwc_q', -1, -1,  -99.0,  209.4,  3.0,  0.5,  10.,  3, 100,
      'exp',  -99.0,  -99.0,  4.0e6,  -99.0,  8.0e-11,  5.0e-3,
      'mie-sphere', 'corPowerLaw_5.0_0.20', -99.0
   ),
])

descriptorFile_gscp2 = np.array([
   (  # cloud liquid water (1-moment mass)
      'cwc_q', 1.0,  1,  -99.0,  -99.0,  -99.0,  -99.0,  -99.0,  3,  1,
      'mono',  -99.0,  -99.0,  -99.0,  -99.0,  2.0e-5,  -99.0,
      'mie-sphere', 'corPowerLaw_24388657.6_2.0', -99.0
   ),
   (  # cloud ice (1-moment mass)
      'iwc_q', 0.2, -1,  -99.0,  130.0,   3.0,  0.684,  2.0,  3,  1,
      'mono_cosmo_ice',  -99.0,  -99.0,  -99.0,  -99.0,  -99.0,  -99.0,
      'ssrg-rt3_0.18_0.89_2.06_0.08', 'corPowerLaw_10.0_0.5', -99.0  
   ),
   (  # rain (1-moment mass)
      'rwc_q', 1.0,  1,  -99.0,  -99.0,  -99.0,  -99.0,  -99.0,  3, 100,
      'exp',  -99.0,  -99.0,  1.0e6,  -99.0,  1.2e-4,  6.0e-3,
      'mie-sphere', 'corPowerLaw_130.0_0.5', -99.0
   ),
   (  # snow (1-moment mass)
      'swc_q', 0.6, -1,  -99.0,  0.038,  2.0,  0.3971,  1.88,  3, 100,
      'exp_cosmo_snow',  -99.0,  -99.0,  -99.0,  -99.0,  5.1e-11,  1.0e-2,
      'ssrg-rt3_0.25_1.00_1.66_0.04', 'corPowerLaw_4.9_0.25', -99.0
   ),
   (  # graupel with N0g=4e6
      'gwc_q', 1.0, -1,  -99.0,  169.6,  3.1,  0.5,  2.5,  3, 100,
      'exp',  -99.0,  -99.0,  4.0e6,  -99.0,  8.0e-11,  5.0e-3,
      'ssrg-rt3_0.30_1.10_1.80_0.05', 'corPowerLaw_5.0_0.20', -99.0
   ),
])

if igscp == 4:
    # set geometry and particle size distribution parameters for Pamtra
    set_pamtra_geo(cloud)
    set_pamtra_geo(rain)
    set_pamtra_geo(ice)
    set_pamtra_geo(snow)
    set_pamtra_geo(graupel)
    set_pamtra_geo(hail)
    print(f"  {cloud.a_geo=}")
    print(f"  {cloud.b_geo=}")
    print(f"  {cloud.a_ms=}")
    print(f"  {cloud.b_ms=}")
    print(f"  {cloud.mu=}")
    print(f"  {cloud.nu=}")
    print(f"  {rain.a_geo=}")
    print(f"  {rain.b_geo=}")
    print(f"  {rain.a_ms=}")
    print(f"  {rain.b_ms=}")
    print(f"  {rain.mu=}")
    print(f"  {rain.nu=}")
    print(f"  {ice.a_geo=}")
    print(f"  {ice.b_geo=}")
    print(f"  {ice.a_ms=}")
    print(f"  {ice.b_ms=}")
    print(f"  {ice.mu=}")
    print(f"  {ice.nu=}")
    print(f"  {snow.a_geo=}")
    print(f"  {snow.b_geo=}")
    print(f"  {snow.a_ms=}")
    print(f"  {snow.b_ms=}")
    print(f"  {snow.mu=}")
    print(f"  {snow.nu=}")
    print(f"  {graupel.a_geo=}")
    print(f"  {graupel.b_geo=}")
    print(f"  {graupel.a_ms=}")
    print(f"  {graupel.b_ms=}")
    print(f"  {graupel.mu=}")
    print(f"  {graupel.nu=}")
    print(f"  {hail.a_geo=}")
    print(f"  {hail.b_geo=}")
    print(f"  {hail.a_ms=}")
    print(f"  {hail.b_ms=}")
    print(f"  {hail.mu=}")
    print(f"  {hail.nu=}")
    #quit()

descriptorFile_gscp4 = np.array([
   (  # cloud liquid water (cwc_q)
      'cwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0, 
      13, 100, 'mgamma', -99.0, -99.0, cloud.nu, cloud.mu, 1.0e-6, 1.0e-4,
      'mie-sphere', 'khvorostyanov01_drops', -99.0
   ),
   (  # cloud ice (iwc_q)
      'iwc_q', 1.0, -1, -99.0, ice.a_ms, ice.b_ms, 0.684, 2.0,
      13, 100, 'mgamma', -99.0, -99.0, ice.nu, ice.mu, 5.0e-6, 1.0e-2,
      'ss-rayleigh-gans_0.18_0.89', 'heymsfield10_particles', -99.0
   ),
   (  # rain water (rwc_q)
      'rwc_q', 1.0, 1, -99.0, -99.0, -99.0, -99.0, -99.0,
      13, 100, 'mgamma', -99.0, -99.0, rain.nu, rain.mu, 80.0e-6, 8.0e-3,
      'mie-sphere', 'khvorostyanov01_drops', -99.0
   ),
   (  # snow (swc_q)
      'swc_q', 1.0, -1, -99.0, snow.a_ms, snow.b_ms, 0.3971, 1.88,
      13, 100, 'mgamma', -99.0, -99.0, snow.nu, snow.mu, 50.0e-6, 8.0e-3,
      'ss-rayleigh-gans_0.18_0.89', 'heymsfield10_particles', -99.0
   ),
   (  # graupel (gwc_q)
      'gwc_q', 1.0, -1, -99.0, graupel.a_ms, graupel.b_ms, -99.0, -99.0,
      13, 100, 'mgamma', -99.0, -99.0, graupel.nu, graupel.mu, 100.0e-6, 8.0e-3,
      'mie-sphere', 'khvorostyanov01_spheres', -99.0
   ),
   (  # hail (hwc_q)
      'hwc_q', 1.0, -1, -99.0, hail.a_ms, hail.b_ms, -99.0, -99.0,
      13, 100, 'mgamma', -99.0, -99.0, hail.nu, hail.mu, 500.0e-6, 8.0e-3,
      'mie-sphere', 'khvorostyanov01_spheres', -99.0
   ),
])

if igscp == 1:
   descriptorFile = descriptorFile_gscp1
elif igscp == 2:
   descriptorFile = descriptorFile_gscp2
elif igscp == 3:
   descriptorFile = descriptorFile_gscp3
elif igscp == 4:
   descriptorFile = descriptorFile_gscp4

# parameter for number concentration in convective clouds
rho_w = 1000.0                 # density of liquid water
rho_g = 400.0                  # density of graupel
t0    = 273.16                 # melting point
n0r   = 8e6                    # intercept of exponential rain PSD
n0g   = 4e6                    # intercept of exponential rain PSD
ascon = 0.069                  # prefactor in m=a*D**2
n0s1  = 13.5 * 5.65e5          # parameter in n0s, Field et al.
n0s2  = 0.107                  # parameter in n0s, Field et al.

# Monitor memory
print_memory_usage("Start Pamtra Script")

# Load data
datapath = datadir+'/'+datestring+'00/'
filename = 'earthcare'+datestring+'.nc'
datafile = datapath+filename
print("Loading "+datafile)
dataf = xr.open_dataset(datafile)
dataf = dataf.isel(zc_idx=slice(0, 100))     # needed to cut some levels due to limitation to 10hPa pressure
dataf = clean_dataset(dataf)                 # remove problematic attributes inherited from ESA file

# Convert datetime64 to float hours and apply maks to reduce dataset
time_hours = ( dataf["time_obs"].dt.hour +
               dataf["time_obs"].dt.minute / 60 +
               dataf["time_obs"].dt.second / 3600)
#mask  = (time_hours >= 18.40) & (time_hours <= 18.50)
#dataf = dataf.sel(time=mask)
print(dataf)

# this removes missing columns 
dataf = dataf.dropna(dim="time", how="any", subset=["time_icon"])

ntfull = dataf.sizes['time']
nbatch = 25000
nstep  = int(ntfull/nbatch)+1

use_subgrid_qc = False
use_subgrid_qi = False
use_subgrid_convection = False

print_memory_usage("Before Pamtra loop")

if nout_start == nout_end:
    nsteps = [nout_start-1]
elif nout_end > nout_start:
    nsteps = range(nout_start-1,nout_end)
else:
    nsteps = range(nstep)

for n in nsteps:
    
    nstart = nbatch*n
    nend   = nbatch*(n+1)             
    nend   = min(nend,ntfull)
    print("Pamtra for n = {}".format(n))
    print("  nstart = {}".format(nstart))
    print("  nend   = {}".format(nend))   
    
    data = dataf.isel(time=slice(nstart, nend))    

    print(data)
    
    if data.sizes["time"] == 0:
        print("Empty batch, skipping")
        continue
        
    Ntimes = data.sizes['time']
    Nhgt   = data.sizes['zc_idx']
    Nhydro = len(descriptorFile)

    print("Read variables for Pamtra")
    times = data.time.values.astype(np.float64) * 1e-9   # time in nanoseconds
    lats  = data.lat.values
    lons  = data.lon.values
    
    zfull = data.zfull.values
    zhalf = data.zhalf.values
    pres  = data.pres.values
    temp  = data.temp.values
    rho   = data.rho.values

    qv = data.qv.values
    qr = data.qr.values
    qi = data.qi.values
    qs = data.qs.values
    qg = np.zeros_like(qs)        # may not be needed, just for safety
    qh = np.zeros_like(qs)        # may not be needed, just for safety
    if use_subgrid_qc:
      qc = data.qc_diag.values
    else:          
      qc = data.qc.values

    nc = np.zeros_like(qc)
    nr = np.zeros_like(qr)
    ni = np.zeros_like(qi)
    ns = np.zeros_like(qs)
    ng = np.zeros_like(qi)
    nh = np.zeros_like(qi)

    if igscp == 3:
      ni = data.ni.values
      # limits for mean mass of cloud ice from ICON code
      mimin = 130*2e-6**3
      mimax = 130*200e-6**3
      ni = np.where( (qi > 0.0) & (qi/ni > mimax), qi/mimax, ni)    
      ni = np.where( (qi > 0.0) & (qi/ni < mimin), qi/mimin, ni)    
    elif igscp == 2:
      qg = data.qg.values
      qg = np.nan_to_num(qg, nan=0.0, posinf=0.0, neginf=0.0) 
    elif igscp == 4:
      qg = data.qg.values
      qh = data.qh.values
      nc = data.nc.values
      nr = data.nr.values
      ni = data.ni.values
      ns = data.ns.values
      ng = data.ng.values
      nh = data.nh.values
      # limit number densities with x_min and x_max of each particle category
      nc = limit_number_density(qc, nc, cloud.x_min, cloud.x_max)
      nr = limit_number_density(qr, nr, rain.x_min, rain.x_max)
      ni = limit_number_density(qi, ni, ice.x_min, ice.x_max)
      ns = limit_number_density(qs, ns, snow.x_min, snow.x_max)
      ng = limit_number_density(qg, ng, graupel.x_min, graupel.x_max)
      nh = limit_number_density(qh, nh, hail.x_min, hail.x_max)

    rh = 100*pyPamtra.meteoSI.q2rh(qv, temp, pres)
    rh = np.where(rh > 100.0, 100.0, rh)               # why and where does this happen?
    rh = np.where(rh <   0.0,   0.0, rh)               #                                        

    print("u:", data.u.shape)
    print("v:", data.v.shape)
    print("u values:", data.u.values.shape)
    print("v values:", data.v.values.shape)

    uv = np.hypot( data.u, data.v ).values
    wh = data.w.values

    ndim  = np.shape(temp)
    ndim2 = np.shape(wh)
    kdim  = ndim[1]
    kdim2 = ndim2[1]-1
    wf  = 0.5*(wh[:,:-1] + wh[:, 1:])                  # average half levels to full levels
    w   = wf[:,kdim2-kdim-1:kdim2-1]                    
    
    print("max pres  = {:.3e}".format(np.max(pres)))
    print("max temp  = {:.3e}".format(np.max(temp)))
    print("max zfull = {:.3e}".format(np.max(zfull)))
    print("max qv = {:.3e}".format(np.max(qv)))
    print("max qc = {:.3e}".format(np.max(qc)))
    print("max qr = {:.3e}".format(np.max(qr)))
    print("max qi = {:.3e}".format(np.max(qi)))
    print("max qs = {:.3e}".format(np.max(qs)))
    print("max qg = {:.3e}".format(np.max(qg)))
    print("max qh = {:.3e}".format(np.max(qh)))

    eps = 1e-15
    if use_subgrid_qi:
        # add subgrid qi from cover_koe similar to treatment in effective radius module of ICON
        qidia = data.qi_diag.values
        qisgs = qidia - qi - qs
        nisgs = np.zeros_like(qisgs)
        qigrd = np.where( (qi > eps), qi, np.nan)                               # avoid divide by zero warning
        qi = qi+qisgs
        if igscp == 3 or igscp == 4:
          nisgs = np.where( (qi > qisgs) & (qi > eps), ni*qisgs/qigrd,nisgs)      # use size information of grid-scale ice
                                                                                  #     if grid-scale ice dominates
          nisgs = np.where( (qisgs > qi) & (qisgs > 0.0), nicooper(temp), nisgs)  # use Cooper curve for sgs ice only
                                                                                  #     if sgs ice dominates
          ni = ni+nisgs
        print("max qi_sgs = {:.3e}".format(np.max(qisgs)))
        print("min qi_sgs = {:.3e}".format(np.min(qisgs)))
        print("max qi_tot = {:.3e}".format(np.max(qi)))
    if use_subgrid_convection:
        # add contribution from convective cores similar to Geer at al. 2007, Appendix B
        qrcon = data.qr_con.values
        qscon = data.qs_con.values
        n0s   = n0s1 * np.exp(-n0s2*np.clip(temp-t0,-40.0,0.0))       # temperature-dependent intercept
        n0s   = np.clip(n0s,1.0e6,1.0e9)                              # limit intercept
        nscon = n0s**(2/3) * (rho*qscon / (2.0 * ascon))**(1/3)/rho   # exponential PSD with m=a*D**2
        nrcon = n0r**(3/4) * (rho*qrcon / (np.pi*rho_w))**(1/4)/rho   # exponential PSD for raindrops
        qs = qs+qscon
        ns = ns+nscon
        qr = qr+qrcon
        nr = nr+nrcon
        print("max qr+qrcon = {:.3e}".format(np.max(qr)))
        print("max qs+qscon = {:.3e}".format(np.max(qs)))

    # add hydrometeors in the stack list as they appear in the descriptorFile
    if igscp == 1 or igscp == 3:
      hydro_q = np.stack([rho*qc, rho*qi, rho*qr, rho*qs, rho*qg], axis=-1)   
      hydro_n = np.stack([rho*nc, rho*ni, rho*nr, rho*ns, rho*ng], axis=-1)      
    elif igscp == 2:
      hydro_q = np.stack([qc, qi, qr, qs, qg], axis=-1)
      hydro_n = np.stack([nc, ni, nr, ns, ng], axis=-1)      
    elif igscp == 4:
      hydro_q = np.stack([qc, qi, qr, qs, qg, qh], axis=-1)
      hydro_n = np.stack([nc, ni, nr, ns, ng, nh], axis=-1)      

    hydro_q[hydro_q < 0.00]  = 0.0                     # remove negative values
    hydro_q[hydro_q > 0.03]  = 0.03                    # upper bound in PAMTRA
   
    hydro_n = np.nan_to_num(hydro_n, nan=1e3, posinf=1e3, neginf=1e3)

    if np.isnan(hydro_q).any():
      print("Error: NaN detected in hydro_q! Exiting.", file=sys.stderr)
      sys.exit(1)
    
    if np.isnan(hydro_n).any():
      print("Error: NaN detected in hydro_n! Exiting.", file=sys.stderr)
      sys.exit(1)
 
    pam = pyPamtra.pyPamtra()
    for hydro in descriptorFile:
        pam.df.addHydrometeor(hydro)
    
    print("Create profiles for Pamtra")
    pam.createProfile(hgt=zfull,
                      press=pres,
                      temp=temp,
                      relhum=rh,
                      hydro_q=hydro_q,
                      hydro_n=hydro_n,
                      wind_uv=uv,
                      wind_w=w,
                      timestamp=times,
                      lat=lats,
                      lon=lons,
                      )

    # Pamtra Settings
    pam.nmlSet["passive"] = False
    pam.nmlSet["active"] = True
    pam.nmlSet["radar_mode"] = 'moments'
    pam.nmlSet["radar_pnoise0"]= -299.99 # Email Davide Ori 20 Nov 2025
    pam.nmlSet["radar_peak_min_snr"] =  -120
    #pam.nmlSet["radar_mode"] = 'simple'
    #pam.set['verbose'] = 0        # set verbosity levels
    pam.set['pyVerbose'] = 1      # change to 0 if you do not want to see job progress number
    pam.p['turb_edr'][:] = 1.0e-4 # If you want to include estimated model turbulence it is possible
    pam.nmlSet['radar_airmotion'] = True
    pam.nmlSet['radar_airmotion_vmin'] = 0.0
    pam.nmlSet['radar_airmotion_model'] = 'constant'
    pam.nmlSet['radar_attenuation'] = 'top-down'
    #pam.nmlSet['radar_attenuation'] = 'disabled'

    # Set the number of spectral averages (n_ave)
    # pam.nmlSet["radar_no_ave"] = 300  # e.g. increase to 300 averages to get more sensitivity

    # run Pamtra
    print("Call Pamtra")
    print_memory_usage("Before ParallelPamtra")
    #pam.runParallelPamtra(np.array([94.0]), pp_deltaX=1000, pp_deltaY=1, pp_deltaF=1, pp_local_workers="auto")
    pam.runParallelPamtra(np.array([94.0]), pp_deltaX=500, pp_deltaY=1, pp_deltaF=1, pp_local_workers=4)
    print_memory_usage("After ParallelPamtra")

    # plot dBZ
    # Ze = pam.r['Ze'][:,:,:,0,0,0]
    # Ze = Ze[0:20000,0,:] 
    # Ze[Ze==-9999] = np.nan
    # mesh = plt.pcolormesh(Ze.T, vmin=-35, vmax=40)
    # plt.colorbar(mesh)
    # plt.savefig(datapath+"plot_Ze_{:03d}.png".format(n+1))
    # plt.savefig("plot_Ze_{:03d}.png".format(n+1))

    # extract relevant data
    Ze = pam.r['Ze'][:,0,:,0,0,0]
    MDV = pam.r['radar_moments'][:,0,:,0,0,0,0]

    # set missing value to np.nan
    Ze[Ze==-9999] = np.nan
    MDV[MDV==-9999] = np.nan

    # add Ze and MDV to xarray dataset
    data = data.assign(Ze=(['time','zc_idx'],Ze))
    data = data.assign(MDV=(['time','zc_idx'],MDV))

    # write netcdf file
    outfile = datapath+"pamtra_{:03d}_v40.nc".format(n+1)
    data.to_netcdf(outfile)
    print("wrote "+outfile)
    print_memory_usage("End of Pamtra Loop")












