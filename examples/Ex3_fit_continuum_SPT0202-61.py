"""
Example 3, to be run in python. This script runs through fitting 
to the 870um continuum data of one source from our ALMA cycle 0 
data, SPT0202-61, which we generated in Example 1. It's a fairly 
complex system, so it uses most all of the capabilities of the code.

Here, we model the field using a standard SIE lens with external
tidal shear, which lenses two source-plane components of the 
emission. There's also an additional unlensed source ~7" south
of the lensed emission detected at ~8-10sigma.

In this case, we actually don't have spectroscopic redshifts for
either the lens or the background source, but the magnification
and source angular size don't actually need those.

The observations were taken at 870um in two different array 
configurations spaced months apart. We fit them jointly, but need
to allow some extra freedom in the model (the absolute flux scale,
for example, doesn't agree to better than the precision of the data).
"""

import numpy as np
import matplotlib.pyplot as pl; pl.ioff()
import sys
sys.path.append('/Users/jspilker/Research/visilens')
import visilens as vl
import time

# We'll use this to help name the output plots
plotfbase = 'SPT0202-61_twosersic'

# These are the files that came out of the first example script,
# you should also have been able to download them from the same place
# you got this script.
comdata = vl.read_visdata('SPT0202-61_com.bin')
extdata = vl.read_visdata('SPT0202-61_ext.bin')

# If you want/need to manipulate the data further, you can do that too:
# comdata.sigma *= 1.5 # really weird data, see what de-influencing it does

# First we define what the lensing environment should look like. For each
# parameter, you must specify an initial guess for its value, can decide
# whether that parameter is fixed or free during fitting, and can impose
# a uniform square prior on its value (ie, it must be between the two values
# in the prior). A value is required, but if you don't specify fixed/free,
# it will not be fixed, and some very loose prior is placed on its value.
# Lens position is relative to the ALMA phase center, with +x west (sorry not
# sorry) and +y north, in arcseconds.
lens = [vl.SIELens(z=0.5,
      x={'value':0.06,'fixed':False,'prior':[-1.5,1.5]},
      y={'value':2.18,'fixed':False,'prior':[1.,3.5]},
      M={'value':1.2e11,'fixed':False,'prior':[1e10,1e12]},
      e=0.4, # will be a free parameter, loose prior
      PA={'value':70.,'prior':[-180.,180.]}), # deg east of north
      
      vl.ExternalShear(
      shear={'value':0.22,'prior':[0.,0.8]},
      shearangle={'value':10.,'prior':[-90.,90.]})]   

# We define two sources which will be lensed by the above (only one
# didn't yield a good fit, as it turns out galaxies don't always look
# like well-behaved Sersic profiles). There's also an unlensed source
# in the field which we model simultaneously. For the lensed sources,
# their positions are relative to the position of the lens we defined
# earlier. For the unlensed source, they're again defined relative to 
# the ALMA pointing center. Flux is in Jy (because our data amplitudes
# are also in Jy), sizes are in arcsec.
src = [vl.SersicSource(z=3.5, lensed=True,
      xoff={'value':-0.1,'fixed':False,'prior':[-1.,1.]},
      yoff={'value':-0.08,'fixed':False,'prior':[-1.,1.]},
      flux={'value':0.003,'fixed':False,'prior':[0.,0.08]},
      reff={'value':0.07,'fixed':False,'prior':[0.,1.]},
      axisratio={'value':0.6,'fixed':False},
      index={'value':0.5,'fixed':False,'prior':[0.1,4.0]},
      PA={'value':90.,'fixed':False,'prior':[0.,180.]}),
      
      vl.SersicSource(z=3.5, lensed=True,
      xoff={'value':0.03,'fixed':False,'prior':[-1.,1.]},
      yoff={'value':0.04,'fixed':False,'prior':[-1.,1.]},
      flux={'value':0.004,'fixed':False,'prior':[0.,0.08]},
      reff={'value':0.25,'fixed':False,'prior':[0.,1.]},
      axisratio={'value':0.7,'fixed':False},
      index={'value':1.,'fixed':False,'prior':[0.2,4.0]},
      PA={'value':10.,'fixed':False,'prior':[-90.,90.]}),

      # Extra unlensed source in the field.
      vl.GaussSource(z=3.5,lensed=False,
      xoff={'value':1.3,'prior':[0.8,1.8]},
      yoff={'value':-3.9,'prior':[-4.4,-3.4]},
      flux=0.003,
      width=0.15)]

 
# These set how we set up the field. In this case, we'll model
# a region within +/- 15" of the phase center (you can check that
# this is sufficient by looking at comdata.uvdist.min()). The
# lensed emission is encompassed within the limits of highresbox,
# ie, between -2.5" and +2.5" in x and -0.5" and +4.5" in y. We
# simulate this region at higher resolution, in this case 0.01".
# The full field is simulated at 0.06". NOTE: the field grid size
# goes to the next-higher power of 2, so fieldres is a coarse limit
# to the actual resolution. In this case, the true resolution will
# be 30/512 = 0.0586", because the grid we asked for gets rounded up
# to 512. Using large grids makes the code take MUCH longer, so I
# strongly recommend choosing the field size and resolution carefully;
# there's a big difference between a 1024x1024 grid and a 512x512 grid!
xmax = 15.
highresbox = [-2.5, 2.5, -0.5, 4.5]
emitres, fieldres = 0.01, 0.06

# Okay, now onto some interferometric wizardry. The data were
# taken using short (~60s) integrations in two tracks separated
# by several months. First, we allow for some variation in the
# absolute flux scaling of the two tracks - they used different
# flux calibrators - and an astrometric shift between the two
# tracks. These can't vary too much from 1 (flux scale), or 0
# (shift), but often are statistically different from those values.
scaleamp =   True # allow flux re-scaling
shiftphase = True # allow astrometric shift

# There's also the possibility that the data have antenna-based
# phase errors which could bias our model results. These could
# be due to imperfectly known antenna positions, or the fact that
# these observations were too short to allow atmospheric variations
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
nwalkers,nburn,nstep = 300,200,200

# For multiprocessing, there's a couple options. For a single machine,
# you can just specify the number of cores to use, like below. It can
# also be run under MPI, in which case you should set mpirun to True,
# and run your script using
# mpirun -np 64 --mca mpi_warn_on_fork 0 python Ex3_fit_continuum_SPT0202-61.py
# [that mpi_warn just turns off some harmless scare text I find annoying]
nthreads = 8
mpirun = False


# Now we run the modeling! This will take awhile, get coffee or something.
# This returns a dictionary which contains all of the chains, magnifications,
# antenna phases, etc., as well as some meta-data about the run like initial
# conditions so you can reproduce a result.
t1 = time.time()
mcmcresult = vl.LensModelMCMC(data=[extdata,comdata],lens=lens,source=src,xmax=xmax,highresbox=highresbox,\
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

# Now we image the data and compare to an image of the best model. The imaging
# is very dumb, so very slow. If you want to do a lot of imaging or more intensive
# imaging, I highly recommend outputting visibilities to uvfits, because I have
# not tried to re-implement all of CASA's imaging capabilities. You can set which/
# how many contours get drawn with the kwargs mapcontours and rescontours (in
# multiples of the noise). In this case, we plot each dataset individually, and
# then both together. You can plot only both of them together by setting
# plotcombined=True, plotall=False. There are some other kwargs you can use to 
# mess with the appearance, check the plot_images docs.
# We get four panels for each dataset (and both together): raw dirty image,
# model dirty image, residuals, and a high-res model of the lensed emission.
f,axarr = vl.plot_images([extdata,comdata],mcmcresult,imsize=300,pixsize=0.1,
      limits=[-6,6,-4,8],mapcontours=np.array([-25,-15,-5,5,15,25,35,45,55,65]),plotall=True)
axarr[2][0].text(0.,-0.3,"Data contours: steps of 10$\sigma$ starting at $\pm$5; "\
            "Residual contours: steps of 1$\sigma$ starting at $\pm$2",\
            transform=axarr[2][0].transAxes)
f.savefig(plotfbase+'_panels.png')
pl.close()

# Create histograms of the modelcal phase offsets. This lets us see if there
# are potentially-bad antennas (consistent, large phase errors) which could
# indicate an antenna position uncertainty or just bad data.
for i in range(2):
      for j in range(mcmcresult['calphases_dset'+str(i)].shape[1]):
            pl.hist(mcmcresult['calphases_dset'+str(i)][:,j]*180/np.pi,bins=50,histtype='step')

      pl.xlabel('Modelcal Phases, dataset {0:.0f}, degrees'.format(i))
      pl.savefig(plotfbase+'_phasesdset{0:.0f}.png'.format(i))
      pl.close()

