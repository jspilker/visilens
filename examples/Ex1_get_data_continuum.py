"""
Example 1, to be run within CASA. This script serves as a guideline
for how to get data out of a CASA ms and into a format which 
visilens can use. We really don't need all that much information,
so we keep only the columns we need.

To keep the number of visibilities low, we first average the data
a bit. In this particular case, the on-source integration times were
only ~60s, so we won't average in time. We will average down each of 
the four ALMA basebands (spectral windows), since this is continuum
data and the fractional bandwidth from the lowest to highest observed
frequency is small. We'll also average the two orthogonal polarizations,
since the source is unpolarized.  Last, for fitting, we need an
accurate estimate of the uncertainty on each visibility. The *relative*
uncertainties in the data are okay, but they're not on any absolute scale,
so we need to calculate what the re-scaling factor should be. To do this,
we take the difference between successive visibilities on each baseline
(these are strong sources, so unfortunately we can't just use the rms)
and re-scale the noise to match. In principle CASA's statwt also does
this, but I found that it sometimes gave bizarre results (some baselines
weighted 100x more than others for no obvious reason, etc.). If you
have better luck with it, feel free to use that instead!
"""

import numpy as np
import os
c = 299792458.0 # in m/s

# Path to the calibrated ms file, and the source name we want.
inms = 'Compact_0202_to_0418.cal.ms'
field = 'SPT0202-61'
spw = '0,1,2,3'

# First we split out just the source we want from our ms file.
outms = field+'_'+inms[:3].lower()+'.ms'
os.system('rm -rf '+outms)
split(inms,outms,field=field,spw=spw,width=128,datacolumn='corrected',
      keepflags=False)

# Now we'll get the visibility columns we need, before manipulating them.
# data_desc_id is a proxy for the spw number.
ms.open(outms,nomodify=True)
visdata = ms.getdata(['uvw','antenna1','antenna2','data','sigma','data_desc_id'])
visdata['data'] = np.squeeze(visdata['data']) # ditch unnecessary extra dimension
ms.close()

# Get the frequencies associated with each spw, because uvw coordinates are in m
tb.open(outms+'/SPECTRAL_WINDOW')
freqs = np.squeeze(tb.getcol('CHAN_FREQ')) # center freq of each spw
tb.close()

# Get the primary beam size from the antenna diameter. Assumes homogeneous array,
#  sorry CARMA users.
tb.open(outms+'/ANTENNA')
diam = np.squeeze(tb.getcol('DISH_DIAMETER'))[0]
PBfwhm = 1.2*(c/np.mean(freqs))/diam * (3600*180/np.pi) # in arcsec
tb.close()

# Data and sigma have both polarizations; average them
visdata['data'] = np.average(visdata['data'],weights=(visdata['sigma']**-2.),axis=0)
visdata['sigma']= np.sum((visdata['sigma']**-2.),axis=0)**-0.5

# Convert uvw coords from m to lambda
for ispw in range(len(spw.split(','))):
      visdata['uvw'][:,visdata['data_desc_id']==ispw] *= freqs[ispw]/c

# Calculate the noise re-scaling, by differencing consecutive visibilities on the
# same baseline. Have to do an ugly double-loop here; would work better if we knew
# in advance how the data were ordered (eg time-sorted). We assume that we can
# re-scale the noise using the mean of the re-scalings from each baseline.
facs = []
for ant1 in np.unique(visdata['antenna1']):
      for ant2 in np.unique(visdata['antenna2']):
            if ant1 < ant2:
                  thisbase = (visdata['antenna1']==ant1) & (visdata['antenna2']==ant2)
                  reals = visdata['data'].real[thisbase]
                  imags = visdata['data'].imag[thisbase]
                  sigs = visdata['sigma'][thisbase]
                  diffrs = reals - np.roll(reals,-1); diffis = imags - np.roll(imags,-1)
                  std = np.mean([diffrs.std(),diffis.std()])
                  facs.append(std/sigs.mean()/np.sqrt(2))

facs = np.asarray(facs); visdata['sigma'] *= facs.mean()
print outms, '| mean rescaling factor: ',facs.mean(), '| rms/beam (mJy): ',1000*((visdata['sigma']**-2).sum())**-0.5


# If we ever want to mess with the data after re-scaling the weights, we have to 
# write them back to the ms file. But, CASA doesn't like that we've averaged
# the polarizations together, so we have to keep them separate for this purpose.
ms.open(outms,nomodify=False)
replace = ms.getdata(['sigma','weight'])
replace['sigma'] *= facs.mean()
replace['weight'] = replace['sigma']**-2.
ms.putdata(replace)
ms.close()

# Create one single array of all this data, then save everything.
allarr = np.vstack((visdata['uvw'][0,:],visdata['uvw'][1,:],visdata['data'].real,
         visdata['data'].imag,visdata['sigma'],visdata['antenna1'],visdata['antenna2']))

outfname = field+'_'+inms[:3].lower()+'.bin'
with open(outfname,'wb')as f:
      allarr.tofile(f)
      f.write(PBfwhm)

