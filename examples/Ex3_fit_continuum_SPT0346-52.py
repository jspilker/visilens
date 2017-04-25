"""
This is a new continuum fitting Example 3 compared to what
used to be up here on github. I swapped out the old one for
this one because this one probably represents a more typical
case for most people, and it's a somewhat simpler system
to model, so the chains converge faster. For reference, I can
run this script on a late-2013 MacBook Pro in ~2 hours.

Example 3, to be run in python. This script fits to some 2mm 
continuum data (140GHz, to be more precise) of SPT0346-52 at
z = 5.656. The primary target of these observations was the
CO(8-7) line, which was detected at high significance. Later on,
in Example 5, I'll show how to do a joint fit to the line and
the continuum.

Here, we model the field using a standard SIE lens with external
tidal shear, which lenses the Sersic profile of emission.

In this case, we actually don't have a spectroscopic redshift for
the lens, but the magnification and source angular size don't 
actually depend on this; only the lens mass changes.
"""

import numpy as np
import matplotlib.pyplot as pl; pl.ioff()
import sys
sys.path.append('/Users/jspilker/Research/visilens')
import visilens as vl
import time

# We'll use this to help name the output plots.
plotfbase = 'SPT0346-52_modelcont'

# This is a file I generated using a script similar to the first
# example. You should also have been able to download it from
# the same place as you got this script.
data = vl.read_visdata('SPT0346-52_cont.bin')

# If you want/need to manipulate the data further, you can do that too:
# data.sigma *= 1.5 # try increasing the noise to map out any local minima

# First we define what the lensing environment should look like. For each
# parameter, you must specify an initial guess for its value, can decide
# whether that parameter is fixed or free during fitting, and can impose
# a uniform square prior on its value (ie, it must be between the two values
# in the prior). A value is required, but if you don't specify fixed/free,
# it will not be fixed, and some very loose prior is placed on its value.
# Lens position is relative to the ALMA phase center, with +x west (sorry not
# sorry) and +y north, in arcseconds.
lens = [
    vl.SIELens(z=0.8,
    x={'value':-0.5,'fixed':False,'prior':[-1.5,0.5]},
    y={'value':0.0,'fixed':False,'prior':[-1.,1.]},
    M={'value':2.8e11,'fixed':False,'prior':[1e9,1e13]},
    e={'value':0.51,'fixed':False,'prior':[0.,0.8]},
    PA={'value':70.9,'fixed':False,'prior':[0,180]}),
    
    vl.ExternalShear(
        shear={'value':0.12,'prior':[0.,0.5]},
        shearangle={'value':122.,'prior':[0.,180.]})]


# We define a Sersic profile source which will be lensed by the above.
# For lensed sources, their positions are relative to the position of 
# the lens we defined earlier. Flux is in Jy (because our data 
# amplitudes are also in Jy), and the source major axis is in arcsec.
# You can add additional source profiles similar to this if the data
# you're modeling aren't well-fit with a single source. You can also
# include "unlensed" sources in the field by setting lensed=False, in
# which case the position is relative to the ALMA phase center.     
source = [
    vl.SersicSource(z=5.6559, lensed=True,
    xoff={'value':0.22,'prior':[-0.8,0.8]},
    yoff={'value':0.27,'prior':[-0.8,0.8]},
    flux={'value':5e-3,'prior':[0.,0.100]}, # 5 mJy
    majax={'value':0.12,'prior':[0.,0.5]},
    index={'value':0.8,'prior':[0.3,2.5]},
    axisratio={'value':0.7,'prior':[0.3,1.]},
    PA={'value':31.,'prior':[0.,180]})]
    
   
# These set how we set up the field. In this case, we'll model
# a region within +/- 12.5" of the phase center (you can check that
# this is sufficient by looking at comdata.uvdist.min()). The
# lensed emission is encompassed within the limits of highresbox,
# ie, between -2.5" and +1.5" in x and -2.0" and +2.0" in y. We
# simulate this region at higher resolution, in this case 0.025".
# The full field is simulated at 0.05". NOTE: the field grid size
# goes to the next-higher power of 2, so fieldres is a coarse limit
# to the actual resolution. In this case, the true resolution will
# be 25/512 = 0.0488", because the grid we asked for gets rounded up
# to 512. Using large grids makes the code take MUCH longer, so I
# strongly recommend choosing the field size and resolution carefully;
# there's a big difference between a 1024x1024 grid and a 512x512 grid! 
# Specify the field and grid parameters
xmax = 12.5
highresbox = [-2.5, +1.5, -2, +2] # xmin, xmax, ymin, ymax
fieldres, emitres = 0.05, 0.025

# Okay, now onto some interferometric wizardry. If your data were
# observed multiple times (eg, multiple executions on ALMA), you
# may need to allow for some variation between the executions in the
# absolute flux scaling of the data and a difference in the 
# astrometry. For example, ALMA claims that the absolute flux scale
# on its data is good to ~<10%, but you can easily have data that
# is more constraining than that; similarly, I've seen data with
# astrometric accuracty good to ~0.2", but the data can constrain
# the lens position far better than this. These data were only observed
# once, so we'll set them to False, but you might need this for
# your data (feel free to email me for help).
scaleamp =   False # do not allow flux re-scaling
shiftphase = False # do not allow astrometric shift

# There's also the possibility that the data have antenna-based
# phase errors which could bias our model results. These could
# be due to imperfectly known antenna positions, or the fact that
# the observations were too short to allow atmospheric variations
# to average over the array. This is sort of like self-calibrating
# your data, except that in this case, we use a model of the field
# generated from the lensing geometry instead of from CLEAN
# components. This lets us marginalize over those phase errors, but
# you can also use this to identify possibly-bad antennas to remove.
# So, DO NOT self-cal your data beforehand! The math behind this
# technique is described in the appendix to Hezaveh et al. (2013),
# 762, 132.
modelcal = True

# Now we do some setup for the MCMC run. These are the usual emcee
# parameters which govern how many chains are run and for how long.
nwalkers,nburn,nstep = 300,300,300

# For multiprocessing, there's a couple options. For a single machine,
# you can just specify the number of cores to use, like below. It can
# also be run under MPI, in which case you should set mpirun to True,
# and run your script using
# mpirun -np 64 --mca mpi_warn_on_fork 0 python Ex3_fit_continuum_SPT0346-52.py
# [that mpi_warn just turns off some harmless scare text I find annoying]
nthreads = 7
mpirun = False


t1 = time.time()
mcmcresult = vl.LensModelMCMC(data=data,lens=lens,source=source,xmax=xmax,highresbox=highresbox,\
      fieldres=fieldres,emitres=emitres,scaleamp=scaleamp,shiftphase=shiftphase,\
      modelcal=modelcal,nwalkers=nwalkers,nburn=nburn,nstep=nstep,nthreads=nthreads,mpirun=mpirun)

t2 = time.time()
print "total time: {0:.1f} hours for {1:.0f} samples".format((t2-t1)/3600.,nwalkers*(nburn+nstep))

# How to save the data is up to you; I find a gzipped python pickle file
# convenient. Can be loaded later with 
# mcmc = pickle.load(gzip.open('chains_'+plotfbase+'.pzip'))
import pickle
import gzip

pickle.dump(mcmcresult,gzip.open('chains_'+plotfbase+'.pzip','wb'))

# Now we make a big parameter covariance triangle plot. This returns a
# matplotlib figure object and array of axes objects, so you can tweak
# to your heart's content.
pl.ioff()
f,axesarray = vl.TrianglePlot_MCMC(mcmcresult,plotmag=True,plotnuisance=True)
f.savefig(plotfbase+'_triangle.png')
pl.close()


# Now we image the data and compare to an image of the best model. The 
# imaging is very dumb, so very slow. If you want to do a lot of imaging 
# or more intensive imaging, I highly recommend outputting visibilities 
# to uvfits, because I have not tried to re-implement all of CASA's imaging
# capabilities. You can set which/how many contours get drawn with the 
# kwargs mapcontours and rescontours (in multiples of the noise). There are
# some other kwargs you can use to mess with the appearance, check the
# plot_images docs. We get four panels for this dataset: raw dirty image,
# model dirty image, residuals, and a high-res model of the lensed emission.
f,axarr = vl.plot_images(data,mcmcresult,imsize=300,pixsize=0.07,
      limits=[-4,4,-4,4],mapcontours=np.array([-25,-5,5,25,45,65,85,105,125]))
axarr[0][0].text(0.,-0.3,"Data contours: steps of 20$\sigma$ starting at $\pm$5; "\
            "Residual contours: steps of 1$\sigma$ starting at $\pm$2",\
            transform=axarr[0][0].transAxes)
f.savefig(plotfbase+'_panels.png')
pl.close()


# Create histograms of the modelcal phase offsets. This lets us see if there
# are potentially-bad antennas (consistent, large phase errors) which could
# indicate an antenna position uncertainty or just bad data. For these data,
# you should find that the antenna phases are all small, <~20deg, which means
# the data are well-calibrated.
for j in range(mcmcresult['calphases_dset0'].shape[1]):
      pl.hist(mcmcresult['calphases_dset0'][:,j]*180/np.pi,bins=50,histtype='step')

pl.xlabel('Modelcal Phases, degrees')
pl.savefig(plotfbase+'_phases.png')
pl.close()

