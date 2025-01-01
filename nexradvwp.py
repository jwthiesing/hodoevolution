"""
Created on 2024.01.26 18:01Z

@author: jwthiesing
Creates and plots a derived vertical wind profile from NEXRAD data
"""

# NOTE: COVARIANCE FILTERING CURRENTLY TURNED OFF

import numpy as np

earth_radius = 6371.0 * 1e3

dndh = 2.724e-5/1e3
k = 1./(1.+(earth_radius*dndh))

def calculate_ground_dist(Reach, eleveach):
	import vwpnew
	return k*earth_radius*np.arcsin((Reach*np.cos(np.deg2rad(eleveach)))/(k*earth_radius + vwpnew.make_heights(Reach, eleveach, mask=np.zeros(Reach.shape)))) # From Rauber textbook

# Filters data arrays
def nexrad_filter_data(R, AZ, EL, VR, params):
	#import read_plot_func as dlplot

	# R is the only one that isn't per-sweep
	eleveach = np.tile(EL,(VR.shape[1],1)).T
	Reach = np.tile(R,(eleveach.shape[0],1))
	eleveach = np.pad(eleveach, ((0,1), (0,0)), mode='edge')
	Reach = np.pad(Reach, ((0,1), (0,0)), mode='edge')
	S = calculate_ground_dist(Reach, eleveach)[:-1]

	new_mask = np.full(VR.shape, False)
	exceeds_dist = np.where(S > params['nexrad_dist_limit'])
	new_mask[exceeds_dist] = True

	VRf = VR
	VRf.mask = new_mask | VRf.mask

	return VRf, new_mask | VRf.mask

# Take in radar object, filter and process data, generate and return VWP
def process_nexrad_vwp(radarobj, params, dealias=True):
	import pyart
	import vwpnew
	from metpy.units import units
	import metpy.calc as mpcalc

	radarfile = radarobj
	
	if dealias:
		radarfile.add_field('velocity', pyart.correct.dealias_region_based(radarfile, interval_splits=5, check_nyquist_uniform=False), replace_existing=True) # Dealias first
	# Load in products sweep by sweep
	azimuth = radarfile.azimuth['data']
	elevation = radarfile.elevation['data']
	AZ = np.empty((radarfile.nsweeps), dtype=np.ndarray)
	EL = np.empty((radarfile.nsweeps), dtype=np.ndarray)
	VR = np.empty((radarfile.nsweeps), dtype=np.ndarray)
	for sweepii in np.arange(0,radarfile.nsweeps,1):
		AZ[sweepii] = azimuth[radarfile.get_slice(sweepii)]
		#AZ[sweepii] = np.append(AZ[sweepii], AZ[sweepii][0])
		EL[sweepii] = elevation[radarfile.get_slice(sweepii)]
		VR[sweepii] = radarfile.fields['velocity']['data'][radarfile.get_slice(sweepii)]

	# Remove dual pol sweeps from data since they lack doppler returns - may come back and change this for data filtering purposes later
	hasvel = []
	for ii, arr in enumerate(VR):
		if np.any(VR[ii]):
			hasvel.append(ii)
	AZ, EL, VR = AZ[hasvel], EL[hasvel], VR[hasvel]
	dop_fixed_el = radarfile.fixed_angle['data'][hasvel]

	# Assign dimension fields to variables
	R = radarfile.range['data']

	# Process VWP for each sweep
	heightmins = np.arange(*params['height_bounds'], params['height_bin_width'])
	levels = heightmins + params['height_bin_width']/2.

	vwp_u = np.empty((dop_fixed_el.shape[0], levels.shape[0]))
	vwp_v = np.empty((dop_fixed_el.shape[0], levels.shape[0]))
	vwp_w = np.empty((dop_fixed_el.shape[0], levels.shape[0]))
	for ii, VRsweep in enumerate(VR):
		VRf, mask = nexrad_filter_data(R=R, AZ=AZ[ii], EL=EL[ii], VR=VRsweep, params=params)
		print(f"Sweep {ii} processing")
		vwp_u[ii, :], vwp_v[ii, :], vwp_w[ii, :], levelsr = vwpnew.derive_vwp(R, AZ[ii], VRf, EL[ii], mask, params)

	# Remove vertical velocity field from low elevation angles (sin(1º) high-skews stuff, etc.)
	full_of_nothing = np.full(levels.size, np.nan)
	w_elevation_cutoff = 3.75 # Determined from qualitative analysis of several fits
	bad_w_elevations = np.where(dop_fixed_el < w_elevation_cutoff)
	vwp_w[bad_w_elevations] = full_of_nothing

	# Count data in each layer for merging
	layer_n_u, layer_n_v, layer_n_w = np.zeros(levels.shape, dtype=int), np.zeros(levels.shape, dtype=int), np.zeros(levels.shape, dtype=int)
	for layer in vwp_u: layer_n_u[np.where(~np.isnan(layer))] += 1
	for layer in vwp_v: layer_n_v[np.where(~np.isnan(layer))] += 1
	for layer in vwp_w: layer_n_w[np.where(~np.isnan(layer))] += 1
	#print(layer_n_u, layer_n_v, layer_n_w)
	
	# Merge data
	merged_vwp_u, merged_vwp_v, merged_vwp_w = np.empty(levels.shape), np.empty(levels.shape), np.empty(levels.shape)
	merged_vwp_u_std, merged_vwp_v_std, merged_vwp_w_std = np.empty(levels.shape), np.empty(levels.shape), np.empty(levels.shape)
	for ii, height in enumerate(levels):
		if layer_n_u[ii] > 1:
			merged_vwp_u[ii] = np.nanmean(vwp_u[:,ii])
			merged_vwp_u_std[ii] = np.nanmean(np.abs(vwp_u[:,ii] - merged_vwp_u[ii]))
		else:
			merged_vwp_u[ii] = np.nan
		if layer_n_v[ii] > 1:
			merged_vwp_v[ii] = np.nanmean(vwp_v[:,ii])
			merged_vwp_v_std[ii] = np.nanmean(np.abs(vwp_v[:,ii] - merged_vwp_v[ii]))
		else:
			merged_vwp_v[ii] = np.nan
		if layer_n_w[ii] > 1:
			merged_vwp_w[ii] = np.nanmean(vwp_w[:,ii])
			merged_vwp_w_std[ii] = np.nanmean(np.abs(vwp_w[:,ii] - merged_vwp_w[ii]))
			#print(np.abs(vwp_w[:,ii] - merged_vwp_w[ii]))
			#print(np.nanmean(np.abs(vwp_w[:,ii] - merged_vwp_w[ii])))
		else:
			merged_vwp_w[ii] = np.nan
	
	merged_vwp_u, merged_vwp_v, merged_vwp_w = merged_vwp_u*-1., merged_vwp_v*-1., merged_vwp_w*-1.

	nonnan = np.where(~np.isnan(merged_vwp_u) & ~np.isnan(merged_vwp_v))[0]
	trimmed_u, trimmed_v, trimmed_levels = merged_vwp_u[nonnan], merged_vwp_v[nonnan], levels[nonnan]

	storm_u, storm_v = np.nan*units('m/s'), np.nan*units('m/s')

	if np.nanmax(trimmed_levels) >= 6000.:
		trimmed_u, trimmed_v, trimmed_levels = trimmed_u*units('m/s'), trimmed_v*units('m/s'), trimmed_levels*units.meters
		storm_u, storm_v = mpcalc.bunkers_storm_motion(mpcalc.height_to_pressure_std(trimmed_levels), trimmed_u, trimmed_v, trimmed_levels)[0]
		if np.isnan(storm_u) or np.isnan(storm_v):
			srh_half, srh_1, srh_3 = np.nan, np.nan, np.nan
		else:
			srh_half = mpcalc.storm_relative_helicity(trimmed_levels, trimmed_u, trimmed_v, 500.*units.meters, storm_u=storm_u, storm_v=storm_v)[2]
			srh_1 = mpcalc.storm_relative_helicity(trimmed_levels, trimmed_u, trimmed_v, 1000.*units.meters, storm_u=storm_u, storm_v=storm_v)[2]
			srh_3 = mpcalc.storm_relative_helicity(trimmed_levels, trimmed_u, trimmed_v, 3000.*units.meters, storm_u=storm_u, storm_v=storm_v)[2]
			print(f"0-0.5km SRH: {srh_half}\n0-1km SRH: {srh_1}\n0-3km SRH: {srh_3}")
	else:
		srh_half, srh_1, srh_3 = np.nan, np.nan, np.nan

	return levels, merged_vwp_u, merged_vwp_v, merged_vwp_w, merged_vwp_u_std, merged_vwp_v_std, merged_vwp_w_std, srh_half, srh_1, srh_3, params, hasvel, storm_u, storm_v