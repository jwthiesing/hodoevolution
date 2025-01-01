"""
Created on 2024.02.21 17:31Z

@author: jwthiesing
"""

#import netCDF4 as nc
import numpy as np
#import metpy.calc as mpcalc
#from metpy.units import units
#import pandas as pd

#np.set_printoptions(threshold=np.inf) # -> Debug code for printing whole arrays (may explode PC)
#np.set_printoptions(precision=5)

earth_radius = 6371.0 * 1e3

dndh = 2.724e-5/1e3
k = 1./(1.+(earth_radius*dndh))

# Returns heights AGL at each gate's range from the lidar
# NOTE: USES CONSTANT REFRACTION, but would take significant BG info to do properly
def make_heights(Reach, elev, mask, instrumentheight=0.0):
	heights = np.sqrt(Reach**2. + (k*earth_radius)**2. + 2.0*Reach*k*earth_radius*np.sin(np.deg2rad(elev))) - k*earth_radius + instrumentheight

	# ***NOTE***: For some reason, numpy keeps rounding these float64 arrays (I checked) to the nearest .0 or .5. I have no idea why, I did not consent to it, but it happens, and it's fairly inconsequential for the final result. Fitting algorithm error is probably significantly larger than a few centimeters' difference in height that's getting binned anyway. Just thought you'd like to know!

	#print(Reach.dtype, elev.dtype)
	#print(type(Reach[0][0]), type())
	#print(Reach, elev)
	#r = 22425.
	#eleva = 0.99
	#height = np.sqrt(r**2. + (k*ae)**2. + 2.0*r*k*ae*np.sin(eleva*np.pi/180.0)) - k*ae + instrumentheight
	#print(heights)
	heights = np.ma.array(heights, mask=mask)
	return heights

# Finds the vertical wind profile from the lidar data
def derive_vwp(R, AZ, VR, elevation, mask, params):
	import pandas as pd
	import read_plot_func as dlplot

	HEIGHTINTERVAL = params['height_bin_width']
	HEIGHTMIN, HEIGHTMAX = params['height_bounds']
	MINSAMPLES = params['minimum_samples_per_bin']

	#elevation = np.pad(elevation, ((0,1)), mode='edge')
	#AZ = np.pad(AZ, ((0,1)), mode='edge')
	#AZ = dlplot.arctheta_to_north0(AZ)
	eleveach = np.tile(elevation,(VR.shape[1],1)).T
	#eleveach = np.pad(eleveach, ((0,0), (0,1)), mode='edge')
	Reach = np.tile(R,(VR.shape[0],1))

	# Create heights array for binning & height ranges
	heights = dlplot.make_heights(Reach, eleveach, mask)
	#hmin = heights.min()
	heightmins = np.arange(HEIGHTMIN, HEIGHTMAX, HEIGHTINTERVAL)
	#heightmins = np.arange(hmin, heights.max(), HEIGHTINTERVAL)
	#heightmins = np.arange(0, heights.max(), HEIGHTINTERVAL) # Use for the u,v vs time plots

	gatesabovemaxh = 0

	# Only using a Series so that the indices can be made the height limits in the future should it be necessary
	# Make lists for each height range of all the data that falls within that height range
	rangevalues = pd.Series(index=np.arange(0,len(heightmins),1), dtype=object)
	for ii in rangevalues.index: rangevalues[ii] = []
	for aa, ray in enumerate(VR):
		for bb, gate in enumerate(ray):
			if not mask[aa][bb]:
				if heights[aa][bb] >= HEIGHTMAX or heights[aa][bb] <= HEIGHTMIN:
					gatesabovemaxh = gatesabovemaxh + 1
				else:
					#hrangeind = int(np.floor((heights[aa][bb]-hmin)/HEIGHTINTERVAL))
					hrangeind = int(np.floor((heights[aa][bb]-HEIGHTMIN)/HEIGHTINTERVAL))
					#print(heights[aa][bb], int(np.floor((heights[aa][bb]-HEIGHTMIN)/HEIGHTINTERVAL)))
					rangevalues[hrangeind].append((gate, AZ[aa], elevation[aa]))
					"""try:
						rangevalues[hrangeind].append((gate, AZ[aa], elevation[aa]))
					except:
						print(heights[aa][bb], hrangeind)"""

	print(f"{gatesabovemaxh} gates found outside height bounds")

	# Cull height ranges to only include ranges that have more than a constant minimum number of samples
	cullind = []
	for ii, vals_in_r in enumerate(rangevalues):
		#print(ii, len(vals_in_r))
		#if len(vals_in_r) >= MINSAMPLES:
		#	cullind.append(ii)
	#rangevalues = np.take(rangevalues,cullind)
	#heightmins = np.take(heightmins,cullind)
		if len(vals_in_r) <= MINSAMPLES:
			rangevalues[ii] = [np.nan]
	#rangevalues.loc[cullind] = 

	heightlims = []
	for height in heightmins:
		heightlims.append((height, height+HEIGHTINTERVAL))

	# This is only seperate so you can make the group sizes logarithmic or exponentially increasing above
	# This is the plotted height of the vector
	heightmids = []
	for limits in heightlims:
		heightmids.append((limits[1]+limits[0])/2.)

	u, v, w = np.zeros(len(heightlims)), np.zeros(len(heightlims)), np.zeros(len(heightlims))
	
	cvs = np.zeros(len(heightmids)) # Covariances
	nss = np.zeros(len(heightmids)) # Number of samples
	# Fit each layer to vr = a + b*cos(theta-thetamax)
	for layii, layer in enumerate(rangevalues):
		if len(layer) > 1: # Tried to use isnan but threw a fit bc it is being passed a list
			val = np.zeros(len(layer), dtype=float)
			az = np.zeros(len(layer), dtype=float)
			elevations = np.zeros(len(layer), dtype=float)
			for ii, element in enumerate(layer): # Unpack tuple array into 3 seperate arrays from the pairs
				val[ii] = element[0]
				az[ii] = element[1]
				elevations[ii] = element[2]

			layu, layv, layw, cv, n_samples = derive_vad_heightlayer(val, az, elevations, params, layii=layii)
			u[layii], v[layii], w[layii] = layu, layv, layw
			cvs[layii] = cv
			nss[layii] = n_samples
			if cv > params['cv_threshold'] and params['filter_by_cv']:
				#u[layii], v[layii], w[layii] = u[layii-1], v[layii-1], w[layii-1]
				u[layii], v[layii], w[layii] = np.nan, np.nan, np.nan
		else:
			u[layii], v[layii], w[layii] = np.nan, np.nan, np.nan

	# Remove unnecessary ending parts of the array, this is solved by setting them to NaN instead
	"""
	if params['filter_by_cv']:
		if np.where(u == u[-1])[0].size > 0:
			lastoriginal = np.where(u == u[-1])[0][0] # Will create problems in EXTREMELY rare cases when the u values in several layers match exactly. They're doubles, so this SHOULD NOT HAPPEN. 
			heightmids = heightmids[:lastoriginal+1]
			heightlims = heightlims[:lastoriginal+1]
			u = u[:lastoriginal+1]
			v = v[:lastoriginal+1]
			w = w[:lastoriginal+1]
	"""
	
	# When refitting, refilter by minimum samples per height bin
	"""
	if params['filter_by_cv']:
		goodvalinds = np.where((nss > params['minimum_samples_per_bin'])*(cvs < params['cv_threshold']))
	else:
		goodvalinds = np.where(nss > params['minimum_samples_per_bin'])
	heightmids = np.array(heightmids)[goodvalinds]
	u = u[goodvalinds]
	v = v[goodvalinds]
	w = w[goodvalinds]
	"""
	underminsamples = np.where((nss < params['minimum_samples_per_bin']))
	u[underminsamples] = np.nan
	v[underminsamples] = np.nan
	w[underminsamples] = np.nan

	# Print layer values, largely for debugging but no real downside
	#for ii, hm in enumerate(heightmids):
		#if cvs[ii] < params['cv_threshold'] and params['filter_by_cv']:
		#	print(ii, cvs[ii], u[ii], v[ii], nss[ii])
		#else:
		#	print(ii, cvs[ii], u[ii], v[ii], nss[ii]) # I think this is just here to be commented out if necessary?

	#u, v = pd.Series(u, index=heightmids), pd.Series(v, index=heightmids)

	if u.size < 1:
		u = np.array([np.nan])
		v = np.array([np.nan])
		w = np.array([np.nan])
		heightmids = np.array([0])

	return u, v, w, np.array(heightmids)

# Fits the layer data to function
def derive_vad_heightlayer(vallay, azlay, elevlay, params, layii=0):
	from scipy import optimize
	import matplotlib.pyplot as plt

	# Fit the function and calculate u and v
	elev = elevlay.mean()
	funcparams, params_covariance = optimize.curve_fit(vr_cos_fit, azlay, vallay, p0=[0.0, 7.5, 180.0], bounds=([-np.inf, 0., 0.], [np.inf, np.inf, 360.]))
	#print(params_covariance[1,1])
	a, b, thetamax = funcparams
	# NOTE: (u, v, w) = (-b*sin(thetamax)/cos(elev), -b*cos(thetamax)/cos())
	layu = -b*np.sin(np.deg2rad(thetamax))/np.cos(np.deg2rad(elev))
	layv = -b*np.cos(np.deg2rad(thetamax))/np.cos(np.deg2rad(elev))
	layw = -a/np.sin(np.deg2rad(elev))

	if params['filter_by_phase_shift']:
		# Shift all data to be referenced relative to thetamax
		azshifted = (azlay-thetamax)%360
		valsshifted = vallay-a
		ind = np.argsort(azshifted)
		azshifted = azshifted[ind]
		valsshifted = valsshifted[ind]

		# Smooth out data for comparison
		smoothspacing = params['smooth_spacing']
		azbegins = np.arange(0, 360., smoothspacing)
		valsmooth = np.empty(azbegins.shape)
		smmask = np.zeros(azbegins.shape)
		smmask[:] = False
		for ii, azbg in enumerate(azbegins):
			valinbin = np.where((azshifted >= azbg) * (azshifted < azbg+smoothspacing))
			if (valinbin[0]).size > 1:
				valsmooth[ii] = np.nanmean(valsshifted[valinbin])
			else:
				smmask[ii] = True
		valsmooth = np.ma.array(valsmooth, mask=smmask)
		h1 = np.where(azbegins < 180.)
		h2 = np.where(azbegins >= 180.)
		valh1s = valsmooth[h1]
		valh2s = valsmooth[h2]
		azhalfsm = np.arange(0, 180., smoothspacing)

		# Calculate error statisrtic
		greater = np.maximum(valh1s, valh2s)
		dev1 = np.abs(valh1s-valh2s) # Difference between the out-of-phase values
		dev2 = np.abs(dev1/greater) # Difference between two values relative to more extreme value
		dev3 = np.abs(dev2-2.) # 2 is ideal factor of difference so substract 2

		# Cull data to redo fit by getting good azimuths first
		goodaz = azbegins
		goodazmask = np.empty(goodaz.shape)
		goodazmask[:] = False
		goodazmask[np.where(dev3 > params['phase_shift_threshold'])] = True
		goodazmask[np.where(dev3 > params['phase_shift_threshold'])[0]+h2[0].size] = True
		goodazmask[np.where(np.ma.getmask(dev3))] = True
		goodazmask[np.where(np.ma.getmask(dev3))[0]+h2[0].size] = True
		goodaz = np.ma.array(goodaz, mask=goodazmask)
		goodaz = (goodaz+thetamax)%360
		ind = np.argsort(goodaz)
		goodaz = goodaz[ind]
		goodazmask = goodazmask[ind]
		#if layii == params['plotfitbin']: print(goodaz)
		azlaynew = np.empty(0)
		vallaynew = np.empty(0)
		for ii, goodazpt in enumerate(goodaz):
			if not goodazmask[ii]:
				goodazind = np.where((azlay >= goodazpt) * (azlay < goodazpt+smoothspacing))
				#if layii == params['plotfitbin']: print(goodazind)
				azlaynew = np.append(azlaynew, azlay[goodazind])
				vallaynew = np.append(vallaynew, vallay[goodazind])
			#else:
				#if layii == params['plotfitbin']: print(goodazpt)
		if vallaynew.size > 0:
			funcparams, params_covariance = optimize.curve_fit(vr_cos_fit, azlaynew, vallaynew, p0=[0.0, 7.5, 180.0], bounds=([-np.inf, 0., 0.], [np.inf, np.inf, 360.]))
			a2, b2, thetamax2 = funcparams
			# NOTE: (u, v, w) = (-b*sin(thetamax)/cos(elev), -b*cos(thetamax)/cos())
			layu = -b2*np.sin(np.deg2rad(thetamax2))/np.cos(np.deg2rad(elev))
			layv = -b2*np.cos(np.deg2rad(thetamax2))/np.cos(np.deg2rad(elev))
			layw = -a2/np.sin(np.deg2rad(elev))
		else:
			a2, b2, thetamax2, layu, layv, layw = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
	
	# Fit plot, for debugging/homogeneity assumption analysis
	if params['plotfit']:
		if layii == params['plotfitbin']: # Plot data & fit in a certain layer
			if params['filter_by_phase_shift']:
				fig, ax3 = plt.subplots(1,4,figsize=(20,5))
				ax3[0].plot(azlay, vallay)
				linerange = np.arange(0, 360, 1.)
				ax3[0].plot(linerange, vr_cos_fit(linerange, a, b, thetamax))
				ax3[0].set_xlim((0,360.))

				# Plot the smoothed half-data
				ax3[1].plot(azhalfsm, valh1s)
				ax3[1].plot(azhalfsm, valh2s)
				ax3[1].set_xlim((0,180.))

				# Plotting error as a function of azimuth
				#ax3[2].plot(azhalfsm, dev1)
				#ax3[2].plot(azhalfsm, dev2)
				ax3[2].plot(azhalfsm, dev3)
				ax3[2].set_xlim((0,180))
				ax3[2].set_ylim((0,params['phase_shift_threshold']*2.))
				#ax3[2].set_ylim((0.,7.5))

				ax3[3].plot(azlaynew, vallaynew)
				ax3[3].plot(linerange, vr_cos_fit(linerange, a2, b2, thetamax2))
				ax3[3].set_xlim((0,360.))
			else:
				fig, ax3 = plt.subplots(1,1)
				ax3.plot(azlay, vallay)
				linerange = np.arange(0, 360, 1.)
				ax3.plot(linerange, vr_cos_fit(linerange, a, b, thetamax))
				ax3.set_xlim((0,360.))
		plt.tight_layout()
	if params['filter_by_phase_shift']: n_samples = azlaynew.size
	else: n_samples = azlay.size
	return layu, layv, layw, params_covariance[1,1], n_samples

# derive_vwp but only one layer, starting at a bottom height and going up
def derive_single_vector(R, AZ, VR, elevation, mask, params):
	import pandas as pd

	MINSAMPLES = params['minimum_samples_per_bin']

	elevation = np.pad(elevation, ((0,1)), mode='edge')
	AZ = np.pad(AZ, ((0,1)), mode='edge')
	#AZ = dlplot.arctheta_to_north0(AZ)
	eleveach = np.tile(elevation,(VR.shape[1],1)).T
	Reach = np.tile(R,(VR.shape[0],1))

	heights = make_heights(Reach, eleveach, mask)
	
	if params['fixed_height'] == 0.:
		FIXED_HEIGHT = heights.min()
	else:
		FIXED_HEIGHT = params['fixed_height']
	FIXED_HEIGHT_DEPTH = params['fixed_height_depth']

	# Make list of all the data that falls within height range
	rangevalues = []
	for aa, ray in enumerate(VR):
		for bb, gate in enumerate(ray):
			if heights[aa][bb] >= FIXED_HEIGHT and heights[aa][bb] <= FIXED_HEIGHT+FIXED_HEIGHT_DEPTH and not mask[aa][bb]:
				rangevalues.append((gate, AZ[aa], elevation[aa]))

	if len(rangevalues) < MINSAMPLES:
		return np.nan, np.nan
	else:
		# Fit layer to vr = a + b*cos(theta-thetamax)
		val = np.zeros(len(rangevalues), dtype=float)
		az = np.zeros(len(rangevalues), dtype=float)
		elevations = np.zeros(len(rangevalues), dtype=float)
		for ii, element in enumerate(rangevalues): # Unpack tuple array into 3 seperate arrays from the pairs
			val[ii] = element[0]
			az[ii] = element[1]
			elevations[ii] = element[2]
	
		u, v, cov, n_samples = derive_vad_heightlayer(val, az, elevations, params, layii=0)

		return u, v

# Fitting function
def vr_cos_fit(x, a, b, thetamax): # Where x is theta or azimuth
	return a + b*np.cos(np.deg2rad(x)-np.deg2rad(thetamax))