### Author: J. W. Thiesing
### Created: 12/31/2024

import datetime as dt
import os, sys
import boto3

LEVEL2_BUCKET = "noaa-nexrad-level2"

# Instantiate S3 client
s3 = boto3.resource('s3')
s3cli = boto3.client('s3')
l2_bucket = s3.Bucket(LEVEL2_BUCKET)

product_parameters_VR = {
	'height_bounds': (0., 10000.), # Maximum and minimum height used, m
	'height_bin_width': 50., # Width of VWP algorithm height bins in m
	'minimum_samples_per_bin': 60, # Minimum total gates for a height bin to not be removed from data (for QC)
	'height_bins_per_cb_tick': 10, # Number of height bins (groups of x size) per height colorbar tick
	'plotfit': False, # Boolean, to plot the fitting algorithm in a specific layer or not
	'plotfitbin': 199, # Layer to plot fitting algorithm of
	'filter_by_cv': True, # Forgoing actual data filtering, this instead filters each layer by the covariance of the fitting function.
	'cv_threshold': 0.15, # Threshold for covariance filtering
	'filter_by_phase_shift': False, # Filter using 180 degree phase shift method
	'phase_shift_threshold': 10., # Phase shift filter threshold
	'smooth_spacing': 2.5, # Spacing of the smoothing/averaging in phase shift filter, 360/smooth_spacing must be even
	'hodograph_grid_inc': 5., # Grid interval of the hodograph plot
	'use_fixed_mesh': True, # Uses refraction and elevation incorporated ground distance rather than just R NOTE: DO NOT TURN OFF!!!!!!
	# NEXRAD SPECIFIC
	'nexrad_dist_limit': 35*1e3, # Maximum ground distance from NEXRAD that returns will be included (m)
	'elev_disp': 19., # Elevation angle it will display the nearest sweep to
}

def rounddaydown(dati):
	return dt.datetime(dati.year, dati.month, dati.day)

# Downloads the RADAR file from S3 bucket, loads into object, returns
def downloadfile(reqtime, site, download_dir='./NEXRAD/'):
	import pyart
	import numpy as np

	directory = f"{dt.datetime.strftime(reqtime, '%Y/%m/%d')}/{site}/"
	reqtime, objectname, tdelta = radartime(reqtime, directory)
	if reqtime == 0:
		return np.NaN, dt.timedelta(days=999)
	else:
		if not os.path.isfile(download_dir+objectname):
			if not os.path.exists(download_dir+'/'.join(objectname.split('/')[0:4])):
				os.makedirs(download_dir+'/'.join(objectname.split('/')[0:4]))
			try:
				with open(download_dir+objectname, 'wb') as f:
					s3cli.download_fileobj(LEVEL2_BUCKET, objectname, f)
			except:
				sys.exit(f"Download or parse failed. Exiting. Object: {objectname}")
		with open(download_dir+objectname, 'rb') as f:
			radarfile = pyart.io.read(f)
		return radarfile

def downloadfiles_knownpath(objectlist, download_dir='./NEXRAD/'):
	import pyart
	import numpy as np

	#directory = f"{dt.datetime.strftime(reqtime, '%Y/%m/%d')}/{site}/"
	#reqtime, objectname, tdelta = radartime(reqtime, directory)
	#if reqtime == 0:
	#	return np.NaN, dt.timedelta(days=999)
	radarfiles = []
	for objectname in objectlist:
		if not os.path.isfile(download_dir+objectname):
			if not os.path.exists(download_dir+'/'.join(objectname.split('/')[0:4])):
				os.makedirs(download_dir+'/'.join(objectname.split('/')[0:4]))
			try:
				with open(download_dir+objectname, 'wb') as f:
					s3cli.download_fileobj(LEVEL2_BUCKET, objectname, f)
			except:
				sys.exit(f"Download or parse failed. Exiting. Object: {objectname}")
		with open(download_dir+objectname, 'rb') as f:
			radarfile = pyart.io.read(f)
		radarfiles.append(radarfile)
	return radarfiles

# Finds the closest radar volume object to requested time
def radartime(reqtime, directory):
	import pandas as pd
	import numpy as np

	dati_list = []
	filename_list = []
	for object in l2_bucket.objects.filter(Prefix=directory):
		if not ('MDM' in object.key or 'NXL2LG' in object.key or 'NXL2SR' in object.key or 'NXL2DP' in object.key):
			dati_list.append(dt.datetime.strptime(object.key.split('/')[4][4:19], '%Y%m%d_%H%M%S'))
			filename_list.append(object.key)
	dati_list = sorted(dati_list)
	filename_list = sorted(filename_list)
	#print(dati_list)
	if pd.Series(dati_list).shape[0] > 0:
		tdelta = abs(pd.Series(dati_list) - reqtime)
		filefound = filename_list[np.where(tdelta == tdelta.min())[0][0]]
		timefound = dati_list[np.where(tdelta == tdelta.min())[0][0]]
		return timefound, filefound, tdelta.min()
	else:
		return 0, 0, dt.timedelta(days=999)

# Finds all radar volume objects between requested times
def radartime_between(start, end, directory):
	import pandas as pd
	import numpy as np

	dati_list = []
	filename_list = []
	for object in l2_bucket.objects.filter(Prefix=directory):
		if not ('MDM' in object.key or 'NXL2LG' in object.key or 'NXL2SR' in object.key or 'NXL2DP' in object.key):
			dati_list.append(dt.datetime.strptime(object.key.split('/')[4][4:19], '%Y%m%d_%H%M%S'))
			filename_list.append(object.key)
	dati_list = sorted(dati_list)
	filename_list = np.array(sorted(filename_list))
	#print(dati_list)
	if pd.Series(dati_list).shape[0] > 0:
		tdeltastart = pd.Series(dati_list) - start
		tdeltaend = pd.Series(dati_list) - end
		filesfound = filename_list[np.where((tdeltastart > dt.timedelta(seconds=0)) & (tdeltaend < dt.timedelta(seconds=0)))[0]]
		#timefound = dati_list[np.where(tdelta == tdelta.min())[0][0]]
		if len(filesfound) == 0: sys.exit(f'No valid files found in time range in {directory}')
		return filesfound
	else:
		sys.exit(f'No files found in {directory}')

# Download ALL volumes between two times
def downloadfiles_betweentimes(starttime, endtime, site, download_dir='./NEXRAD/'):
	# To split it into several days, get the day component from the dati object and add a day and treat that as the end point for a given day.
	# Continue doing this until the end of the day is a greater time value than the overall end of the requested period.
	days = []
	nextday = rounddaydown(starttime) + dt.timedelta(days=1)
	if nextday < endtime:
		while nextday < endtime:
			nextday = rounddaydown(starttime) + dt.timedelta(days=1)
			if nextday >= endtime: break
			days.append((starttime,nextday))
			starttime = nextday
	days.append((starttime,endtime))

	for period in days:
		start, end = period
		directory = f"{dt.datetime.strftime(start, '%Y/%m/%d')}/{site}/"
		filesfound = radartime_between(start, end, directory)
		radarfiles = downloadfiles_knownpath(filesfound)

	return radarfiles

def getradarfiles(start, end, site, interval):
	if interval == 0: 
		radarfiles = downloadfiles_betweentimes(start, end, site)
	else:
		radarfiles = []
		reqtime = start
		while reqtime < end:
			radarfiles.append(downloadfile(reqtime, site))
			reqtime = reqtime + dt.timedelta(minutes=interval)

	return radarfiles

def gethodos(radarfiles):
	import nexradvwp

	hodos, hodotimes = [], []
	for radarfile in radarfiles:
		if not site[0] == "T":
			levels, u, v, w, u_std, v_std, w_std, srh_half, srh_1, srh_3, params, hasvel, storm_u, storm_v = nexradvwp.process_nexrad_vwp(radarfile, product_parameters_VR, dealias=True)
		else:
			levels, u, v, w, u_std, v_std, w_std, srh_half, srh_1, srh_3, params, hasvel, storm_u, storm_v = nexradvwp.process_nexrad_vwp(radarfile, product_parameters_VR, dealias=False)
		hodo = {
			'levels': levels,
			'u': u,
			'v': v,
		}
		hodos.append(hodo)
		hodotimes.append(dt.datetime.strptime(radarfile.time['units'][14:], '%Y-%m-%dT%H:%M:%SZ'))

	return hodos, hodotimes

def drawhodos(hodos, hodotimes, site):
	import metpy.plots as mplots
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	import matplotlib as mpl
	import numpy as np
	plt.rcParams.update({'font.size': 14})

	fig, ax = plt.subplots(1,1,figsize=[8,8],gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.05, 'top': 0.95})

	start = hodotimes[0]
	end = hodotimes[-1]
	totalseconds = (end - start).total_seconds()
	cmap = mpl.colormaps['plasma']
	mag = lambda u, v: (u**2 + v**2)**0.5

	maxspd = 0.0
	for hodo in hodos:
		for u, v in zip(hodo['u'], hodo['v']):
			if mag(u,v) > maxspd: maxspd = mag(u,v)


	h = mplots.Hodograph(ax, component_range=maxspd*1.25)
	for hodo, time in zip(hodos, hodotimes):
		hodo['levels'] = np.array(hodo['levels'])
		sincestart = (time - start).total_seconds()
		color = mpl.colors.rgb_to_hsv(cmap(sincestart/totalseconds)[:-1])
		colors = np.full((len(hodo['levels']),3), color)
		colors[:,1] = (colors[:,1] * ((np.nanmax(hodo['levels']) - hodo['levels'])/np.nanmax(hodo['levels'])))
		colorstrings = []
		for color in colors:
			colorstrings.append(mpl.colors.to_hex(mpl.colors.hsv_to_rgb(color)))
		#print(colorstrings)
		hplot = h.plot_colormapped(hodo['u'], hodo['v'], hodo['levels'], intervals=hodo['levels'], colors=colorstrings)
	
	h.add_grid(increment=product_parameters_VR['hodograph_grid_inc'])
	ax.set_xlabel("u (ms^-1)")
	ax.set_ylabel("v (ms^-1)")
	ax.set_title(f"NEXRAD-Derived Hodographs, {site}, {start} to {end}")

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="5%", pad=0.05, axes_class=plt.Axes)
	ticklabels = []
	curr = start
	while curr <= end:
		ticklabels.append(curr.strftime('%Y-%m-%d %H:%M'))
		num_ticks = 4
		curr = curr + dt.timedelta(seconds=totalseconds/num_ticks)
	cbar=plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap=cmap),ticks=np.arange(0,1+num_ticks**-1,num_ticks**-1),pad=0.065,cax=cax,orientation='horizontal')
	cbar.ax.tick_params(labelsize=12)
	cbar.ax.set_xticklabels(ticklabels)
	#cbar.set_label("Time", rotation=0,fontsize=12)

	plt.tight_layout()
	plt.show()

	return

if __name__ == "__main__":
	import argparse, pickle
	# Parse arguments!!
	parser = argparse.ArgumentParser(
	                    prog='hodoevolution.py',
	                    description='Automatically downloads and displays NEXRAD-derived hodographs for a given time range and interval',
	                    epilog='')
	parser.add_argument('startdati', help='Start of requested period, YYYYMMDD-HHMMSS')
	parser.add_argument('enddati', help='End of requested period, YYYYMMDD-HHMMSS')
	parser.add_argument('interval', help='Time interval between used volumes, minutes. To get every volume in the requested period, set this to 0.')
	parser.add_argument('site', help='NEXRAD site')
	parser.add_argument('-rfp', default=False, dest='rfp', action='store_true')
	args = parser.parse_args(sys.argv[1:])
	site = args.site.upper()

	if args.rfp:
		with open('./latest.pickle', 'rb') as f:
			hodos, hodotimes = pickle.load(f)
	else:
		start = dt.datetime.strptime(args.startdati, "%Y%m%d-%H%M%S")
		end = dt.datetime.strptime(args.enddati, "%Y%m%d-%H%M%S")
		interval = int(args.interval)
	
		radarfiles = getradarfiles(start, end, site, interval)
		hodos, hodotimes = gethodos(radarfiles)
		with open(f'./latest.pickle', 'wb') as f:
			pickle.dump((hodos, hodotimes), f)
	drawhodos(hodos, hodotimes, site)