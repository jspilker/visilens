"""
Example 4, to be run in python. In this example, we fit the CO
emission using the data made from Ex2 (which you should be
able to download from the same place you got this script).

For these data, I just fixed the lens properties to those
we got from the ALMA 870um data, because the ALMA data have
way higher signal-to-noise than these do. You could also do
one big joint fit, if you needed to.
"""

import numpy as np
import matplotlib.pyplot as pl; pl.ioff()
import sys
sys.path.append('/Users/jspilker/Research/visilens')
import visilens as vl
import time
import glob

# We match the file names made by our Example 2 script.
center = '0'
width = '200'

# Read in the data from each day's observations...
datasets = []
for f in sorted(glob.glob('spt0346linedata/spt0346_*_'+center+'_dv'+width+'*.bin')):
      datasets.append(vl.read_visdata(f))
      
# and concatenate it all together. We could have done this earlier when
# we split out the data, but this way we keep open the option of re-scaling the
# amplitudes on different days if we ever have enough s/n for that.
atcadata = vl.concatvis(datasets)
atcadata.filename = 'All, '+center+'km/s, '+width+'km/s wide'      

# A plot name helper
plotfbase = 'SPT0346-52_CO21_'+center+'_dv'+width


# We set the lens parameters to the ALMA best-fit values.
# Lens positions are relative to the  pointing center offset
# ALMA pointed at 03:46:41.19, -52:05:05.5, ATCA at 03:46:41.13, -52:05:02.1
# to calculate offset, use astropy.coordinates & astropy.units:
# import astropy.units as u; import astropy.coordinates as ac
# alma = ac.SkyCoord('03h46m41.19s','-52d05m05.5s')
# atca = ac.SkyCoord('03h46m41.13s','-52d05m02.1s')
# x = xLens*u.arcsec - (alma.ra - atca.ra).to(u.arcsec)*np.cos(alma.dec)
# y = (alma.dec + yLens*u.arcsec - atca.dec).to(u.arcsec)

# Best-fit lens to combined data, sersic + shear
# ALMA xL = 0.806, yL = 3.036 --> xL_atca = 0.253, yL_atca = -0.364
lens = [vl.SIELens(z=0.8,
      x={'value':0.253,'fixed':True,'prior':[-0.221,0.779]},
      y={'value':-0.364,'fixed':True,'prior':[-0.828,0.172]},
      M={'value':2.811e11,'fixed':True,'prior':[1e10,5e13]},
      e={'value':0.515,'fixed':True,'prior':[0.4,0.6]},
      PA={'value':70.90,'fixed':True,'prior':[60, 95]}),
      
      vl.ExternalShear(
      shear={'value':0.119,'fixed':True},
      shearangle={'value':122.13,'fixed':True})]

# We're just going to model the source as a simple symmetric Gaussian
src  = vl.GaussSource(z=5.65,
      xoff={'value':0.142,'fixed':False,'prior':[-1., 1.]},
      yoff={'value':0.30,'fixed':False,'prior':[-1., 1.]},
      flux={'value':0.000273,'fixed':False,'prior':[0.,0.002]},
      width={'value':0.108,'fixed':False,'prior':[0.02,0.5]})


# Set up the gridding for the moedling.
# The ATCA primary beam is ~70arcsec, but a 38arcsec box keeps 
# our grid to 512x512 instead of 1024x1024, and our shortest 
# baselines don't care about this anyway.
xmax = 38.
highresbox = [0.3-2.5, 0.3+2.5, -0.36-2.5, -0.36+2.5]
emitres, fieldres = 0.03, 0.15

# In this case, we don't have much signal-to-noise, so we won't
# allow flux scaling or astrometric shifts
scaleamp, shiftphase = False, False

# Similarly, we don't allow antenna-based phase errors. If these
# were significant, we'd see them in the delays across the ATCA
# bandwidth. Also, we observed many-hour tracks, so atmospheric
# effects have largely averaged out.
modelcal = False

# Do some setup for the MCMC things and multiprocessing
nwalkers,nburn,nstep = 300,200,200
nthreads=4 # use 4 cores for the calculations
mpirun = False

# Now we do the actual MCMC calculations, as in the previous example
t1 = time.time()
mcmcresult = vl.LensModelMCMC(atcadata,lens,src,xmax=xmax,highresbox=highresbox,\
      fieldres=fieldres,emitres=emitres,scaleamp=scaleamp,shiftphase=shiftphase,\
      modelcal=modelcal,nwalkers=nwalkers,nburn=nburn,nstep=nstep,nthreads=nthreads,pool=None,mpirun=mpirun)

t2 = time.time()
print "total time: {0:.1f} hours for {1:.0f} samples".format((t2-t1)/3600.,nwalkers*(nburn+nstep))

# We just dump out the results of the mcmc run.
import pickle
import gzip
pickle.dump(mcmcresult,gzip.open('chains_'+plotfbase+'.pzip','w'))

# Plot triangle degeneracy plot.
f,axesarray = vl.TrianglePlot_MCMC(mcmcresult,plotmag=True,plotnuisance=True)
f.savefig(plotfbase+'_triangle.png')
pl.close()


# Plot images, and we're done!
if plotpanels:
      f,axarr = vl.plot_images(atcadata,mcmcresult,imsize=500,pixsize=0.07,
            limits=[-5,5,-5,5],mapcontours=np.array([-4,4,6,8,10]))
      axarr[0][0].text(0.,-0.2,"Data contours: steps of 2$\sigma$ starting at $\pm$4; "\
            "Residual contours: Steps of 1$\sigma$ starting at $\pm$2",\
            transform=axarr[0][0].transAxes)
      f.savefig(plotfbase+'_panels.png')
      pl.close()
