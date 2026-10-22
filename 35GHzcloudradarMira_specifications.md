#35-GHz cloud radar Mira-35 NMRA (TROPOS, LACROS)
#radar_no_ave should be set in such a way that the temporal resolution is approx. 2.8 s
#radar_pnoise0: Noise level reflectivity factor at 1000 m height in dBZ. Determined from Invercargill observation 2026/01/06, 03:00-06:00 UTC

pam.nmlSet['radar_fwhr_beamwidth_deg']=0.5

pam.nmlSet['radar_integration_time']=2.8

pam.nmlSet['radar_max_v']=6.255

pam.nmlSet['radar_min_v']=-6.255

pam.nmlSet['radar_nfft']=512

pam.nmlSet['radar_no_ave']=30

pam.nmlSet['radar_pnoise0']=-56

pam.runParallelPamtra(np.array([35.5]), pp_deltaX=1, pp_deltaY=1, pp_deltaF=1, pp_local_workers=cores)

pam.writeResultsToNetCDF('data/20151119'+pam.nmlSet["radar_mode"][:3]+'_Mira35_NMRA_LACROS.nc')

