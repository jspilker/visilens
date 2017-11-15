import numpy as np
import scipy.sparse
import os
import sys
import emcee
import copy
from astropy.cosmology import Planck15
from class_utils import *
from lensing import *
from utils import *
from calc_likelihood import calc_vis_lnlike

arcsec2rad = np.pi/180/3600

def LensModelMCMC(data,lens,source,
                  xmax=30.,highresbox=[-3.,3.,-3.,3.],emitres=None,fieldres=None,
                  sourcedatamap=None, scaleamp=False, shiftphase=False,
                  modelcal=True,cosmo=Planck15,
                  nwalkers=1e3,nburn=1e3,nstep=1e3,pool=None,nthreads=1,mpirun=False):
      """
      Wrapper function which basically takes what the user wants and turns it into the
      format needed for the acutal MCMC lens modeling.
      
      Inputs:
      data:
            One or more visdata objects; if multiple datasets are being
            fit to, should be a list of visdata objects.
      lens:
            Any of the currently implemented lens objects or ExternalShear.
      source:
            One or more of the currently implemented source objects; if more than
            one source to be fit, should be a list of multiple sources.
      xmax:
            (Half-)Grid size, in arcseconds; the grid will span +/-xmax in x&y
      highresbox:
            The region to model at higher resolution (to account for high-magnification
            and differential lensing effects), as [xmin, xmax, ymin, ymax]. 
            Note the sign convention is: +x = West, +y = North, like the lens
            positions.
      sourcedatamap:
            A list of length the number of datasets which tells which source(s)
            are to be fit to which dataset(s). Eg, if two sources are to be fit
            to two datasets jointly, should be [[0,1],[0,1]]. If we have four
            sources and three datasets, could be [[0,1],[0,1],[2,3]] to say that the
            first two sources should both be fit to the first two datasets, while the
            second two should be fit to the third dataset. If None, will assume
            all sources should be fit to all datasets.
      scaleamp:
            A list of length the number of datasets which tells whether a flux
            rescaling is allowed and which dataset the scaling should be relative to.
            False indicates no scaling should be done, while True indicates that
            amplitude scaling should be allowed.
      shiftphase:
            Similar to scaleamp above, but allowing for positional/astrometric offsets.
      modelcal:
            Whether or not to perform the pseudo-selfcal procedure of H+13
      cosmo:
            The cosmology to use, as an astropy object, e.g.,
            from astropy.cosmology import WMAP9; cosmo=WMAP9
            Default is Planck15.
      nwalkers:
            Number of walkers to use in the mcmc process; see dan.iel.fm/emcee/current
            for more details.
      nburn:
            Number of burn-in steps to take with the chain.
      nstep:
            Number of actual steps to take in the mcmc chains after the burn-in
      nthreads:
            Number of threads (read: cores) to use during the fitting, default 1.
      mpirun:
            Whether to parallelize using MPI instead of multiprocessing. If True,
            nthreads has no effect, and your script should be run with, eg,
            mpirun -np 16 python lensmodel.py.

      Returns:
      mcmcresult:
            A nested dict containing the chains requested. Will have all the MCMC
            chain results, plus metadata about the run (initial params, data used,
            etc.). Formatting still a work in progress (esp. for modelcal phases).
      chains:
            The raw chain data, for testing.
      blobs:
            Everything else returned by the likelihood function; will have
            magnifications and any modelcal phase offsets at each step; eventually
            will remove this once get everything packaged up for mcmcresult nicely.
      colnames:
            Basically all the keys to the mcmcresult dict; eventually won't need
            to return this once mcmcresult is packaged up nicely.
      """

      if pool: nthreads = 1
      elif mpirun:
            nthreads = 1
            from emcee.utils import MPIPool
            pool = MPIPool(debug=False,loadbalance=True)
            if not pool.is_master():
            	pool.wait()
            	sys.exit(0)
      else: pool = None

      # Making these lists just makes later stuff easier since we now know the dtype
      lens = list(np.array([lens]).flatten())
      source = list(np.array([source]).flatten()) # Ensure source(s) are a list
      data = list(np.array([data]).flatten())     # Same for dataset(s)
      scaleamp = list(np.array([scaleamp]).flatten())
      shiftphase = list(np.array([shiftphase]).flatten())
      modelcal = list(np.array([modelcal]).flatten())
      if len(scaleamp)==1 and len(scaleamp)<len(data): scaleamp *= len(data)
      if len(shiftphase)==1 and len(shiftphase)<len(data): shiftphase *= len(data)
      if len(modelcal)==1 and len(modelcal)<len(data): modelcal *= len(data)
      if sourcedatamap is None: sourcedatamap = [None]*len(data)

      # emcee isn't very flexible in terms of how it gets initialized; start by
      # assembling the user-provided info into a form it likes
      ndim, p0, colnames = 0, [], []
      # Lens(es) first
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__=='SIELens':
                  for key in ['x','y','M','e','PA']:
                        if not vars(ilens)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(ilens)[key]['value'])
                              colnames.append(key+'L'+str(i))
            elif ilens.__class__.__name__=='ExternalShear':
                  for key in ['shear','shearangle']:
                        if not vars(ilens)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(ilens)[key]['value'])
                              colnames.append(key)
      # Then source(s)
      for i,src in enumerate(source):
            if src.__class__.__name__=='GaussSource':
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(src)[key]['value'])
                              colnames.append(key+'S'+str(i))
            elif src.__class__.__name__=='SersicSource':
                  for key in ['xoff','yoff','flux','majax','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(src)[key]['value'])
                              colnames.append(key+'S'+str(i))
            elif src.__class__.__name__=='PointSource':
                  for key in ['xoff','yoff','flux']:
                        if not vars(src)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(src)[key]['value'])
                              colnames.append(key+'S'+str(i))
      # Then flux rescaling; only matters if >1 dataset
      for i,t in enumerate(scaleamp[1:]):
            if t:
                  ndim += 1
                  p0.append(1.) # Assume 1.0 scale factor to start
                  colnames.append('ampscale_dset'+str(i+1))
      # Then phase/astrometric shift; each has two vals for a shift in x&y
      for i,t in enumerate(shiftphase[1:]):
            if t:
                  ndim += 2
                  p0.append(0.); p0.append(0.) # Assume zero initial offset
                  colnames.append('astromshift_x_dset'+str(i+1))
                  colnames.append('astromshift_y_dset'+str(i+1))

      # Get any model-cal parameters set up. The process involves some expensive
      # matrix inversions, but these only need to be done once, so we'll do them
      # now and pass the results as arguments to the likelihood function. See docs
      # in calc_likelihood.model_cal for more info.
      for i,dset in enumerate(data):
            if modelcal[i]:
                  uniqant = np.unique(np.asarray([dset.ant1,dset.ant2]).flatten())
                  dPhi_dphi = np.zeros((uniqant.size-1,dset.u.size))
                  for j in range(1,uniqant.size):
                        dPhi_dphi[j-1,:]=(dset.ant1==uniqant[j])-1*(dset.ant2==uniqant[j])
                  C = scipy.sparse.diags((dset.sigma/dset.amp)**-2.,0)
                  F = np.dot(dPhi_dphi,C*dPhi_dphi.T)
                  Finv = np.linalg.inv(F)
                  FdPC = np.dot(-Finv,dPhi_dphi*C)
                  modelcal[i] = [dPhi_dphi,FdPC]


      # Create our lensing grid coordinates now, since those shouldn't be
      # recalculated with every call to the likelihood function
      xmap,ymap,xemit,yemit,indices = GenerateLensingGrid(data,xmax,highresbox,
                                                fieldres,emitres)

      # Calculate the uv coordinates we'll interpolate onto; only need to calculate
      # this once, so do it here.
      kmax = 0.5/((xmap[0,1]-xmap[0,0])*arcsec2rad)
      ug = np.linspace(-kmax,kmax,xmap.shape[0])

      # Calculate some distances; we only need to calculate these once.
      # This assumes multiple sources are all at same z; should be this
      # way anyway or else we'd have to deal with multiple lensing planes
      if cosmo is None: cosmo = Planck15
      Dd = cosmo.angular_diameter_distance(lens[0].z).value
      Ds = cosmo.angular_diameter_distance(source[0].z).value
      Dds= cosmo.angular_diameter_distance_z1z2(lens[0].z,source[0].z).value

      p0 = np.array(p0)
      # Create a ball of starting points for the walkers, gaussian ball of 
      # 10% width; if initial value is 0 (eg, astrometric shift), give a small sigma
      # for angles, generally need more spread than 10% to sample well, do 30% for those cases [~0.5% >180deg for p0=100deg]
      isangle = np.array([0.30 if 'PA' in s or 'angle' in s else 0.1 for s in colnames])
      initials = emcee.utils.sample_ball(p0,np.asarray([isangle[i]*x if x else 0.05 for i,x in enumerate(p0)]),int(nwalkers))

      # All the lens objects know if their parameters have been altered since the last time
      # we calculated the deflections. If all the lens pars are fixed, we only need to do the
      # deflections once. This step ensures that the lens object we create the sampler with
      # has these initial deflections.
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__ == 'SIELens': ilens.deflect(xemit,yemit,Dd,Ds,Dds)
            elif ilens.__class__.__name__ == 'ExternalShear': ilens.deflect(xemit,yemit,lens[0])

      # Create the sampler object; uses calc_likelihood function defined elsewhere
      lenssampler = emcee.EnsembleSampler(nwalkers,ndim,calc_vis_lnlike,
            args = [data,lens,source,Dd,Ds,Dds,ug,
                    xmap,ymap,xemit,yemit,indices,
                    sourcedatamap,scaleamp,shiftphase,modelcal],
            threads=nthreads,pool=pool)

      
      # Run burn-in phase
      print "Running burn-in... "
      #pos,prob,rstate,mus = lenssampler.run_mcmc(initials,nburn,storechain=False)
      for i,result in enumerate(lenssampler.sample(initials,iterations=nburn,storechain=False)):
            if i%20==0: print 'Burn-in step ',i,'/',nburn
            pos,prob,rstate,blob = result
      
      
      lenssampler.reset()
      
      # Run actual chains
      print "Done. Running chains... "
      for i,result in enumerate(lenssampler.sample(pos,rstate0=rstate,iterations=nstep,storechain=True)):
            if i%20==0: print 'Chain step ',i,'/',nstep
      
      #lenssampler.run_mcmc(pos,nstep,rstate0=rstate)
      if mpirun: pool.close()
      print "Mean acceptance fraction: ",np.mean(lenssampler.acceptance_fraction)

      #return lenssampler.flatchain,lenssampler.blobs,colnames
      
      # Package up the magnifications and modelcal phases; disregards nan points (where
      # we failed the prior, usu. because a periodic angle wrapped).
      blobs = lenssampler.blobs
      mus = np.asarray([[a[0] for a in l] for l in blobs]).flatten(order='F')
      bad = np.where(np.asarray([np.any(np.isnan(m)) for m in mus],dtype=bool))[0]
      for k in bad: mus[k] = np.array([np.nan]*len(source))
      mus = np.asarray(list(mus),dtype=float).reshape((-1,len(source)),order='F') # stupid-ass hack
      bad = np.isnan(mus)[:,0]
      #bad = bad.reshape((-1,len(source)),order='F')[:,0]
      #mus = np.atleast_2d(np.asarray([mus[i] if not bad[i] else [np.nan]*len(source) for i in range(mus.size)])).T
      colnames.extend(['mu{0:.0f}'.format(i) for i in range(len(source))])

      
      # Assemble the output. Want to return something that contains both the MCMC chains
      # themselves, but also metadata about the run.
      mcmcresult = {}

      # keep track of git revision, for reproducibility's sake
      # if run under mpi, this will spew some scaremongering warning text,
      # but it's fine. use --mca mpi_warn_on_fork 0 in the mpirun statement to disable
      try: 
            import subprocess
            gitd = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
            mcmcresult['githash'] = subprocess.check_output('git --git-dir={0:s} --work-tree={1:s} '\
                  'rev-parse HEAD'.format(gitd+'/.git',gitd),shell=True).rstrip()
      except:
            mcmcresult['githash'] = 'No repo found'
      
      
      mcmcresult['datasets'] = [dset.filename for dset in data] # Data files used

      mcmcresult['lens_p0'] = lens      # Initial params for lens,src(s),shear; also tells if fixed, priors, etc.
      mcmcresult['source_p0'] = source
      
      if sourcedatamap: mcmcresult['sourcedatamap'] = sourcedatamap
      mcmcresult['xmax'] = xmax
      mcmcresult['highresbox'] = highresbox
      mcmcresult['fieldres'] = fieldres
      mcmcresult['emitres'] = emitres
      if any(scaleamp): mcmcresult['scaleamp'] = scaleamp
      if any(shiftphase): mcmcresult['shiftphase'] = shiftphase

      mcmcresult['chains'] = np.core.records.fromarrays(np.hstack((lenssampler.flatchain[~bad],mus[~bad])).T,names=colnames)
      mcmcresult['lnlike'] = lenssampler.flatlnprobability[~bad]
      
      # Keep track of best-fit params, derived from chains.
      c = copy.deepcopy(mcmcresult['chains'])
      mcmcresult['best-fit'] = {}
      pbest = []
      # Calculate the best fit values as medians of each param
      lens,source = copy.deepcopy(mcmcresult['lens_p0']), copy.deepcopy(mcmcresult['source_p0'])
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__ == 'SIELens':
                  ilens.__dict__['_altered'] = True
                  for key in ['x','y','M','e','PA']:
                        if not vars(ilens)[key]['fixed']:
                              ilens.__dict__[key]['value'] = np.median(c[key+'L'+str(i)])
                              pbest.append(np.median(c[key+'L'+str(i)]))
            elif ilens.__class__.__name__ == 'ExternalShear':
                  for key in ['shear','shearangle']:
                        if not vars(ilens)[key]['fixed']:
                              ilens.__dict__[key]['value'] = np.median(c[key])
                              pbest.append(np.median(c[key]))
      
      mcmcresult['best-fit']['lens'] = lens

      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if src.__class__.__name__ == 'GaussSource':
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
                              pbest.append(np.median(c[key+'S'+str(i)]))
            elif src.__class__.__name__ == 'SersicSource':
                  for key in ['xoff','yoff','flux','majax','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
                              pbest.append(np.median(c[key+'S'+str(i)]))
            elif src.__class__.__name__ == 'PointSource':
                  for key in ['xoff','yoff','flux']:
                        if not vars(src)[key]['fixed']:
                              src.__dict__[key]['value'] = np.median(c[key+'S'+str(i)])
                              pbest.append(np.median(c[key+'S'+str(i)]))

      mcmcresult['best-fit']['source'] = source
      mcmcresult['best-fit']['magnification'] = np.median(mus[~bad],axis=0)

      # Any amplitude scaling or astrometric shifts
      bfscaleamp = np.ones(len(data))
      if 'scaleamp' in mcmcresult.keys():
            for i,t in enumerate(mcmcresult['scaleamp']): # only matters if >1 datasets
                  if i==0: pass
                  elif t: 
                        bfscaleamp[i] = np.median(c['ampscale_dset'+str(i)])
                        pbest.append(np.median(c['ampscale_dset'+str(i)]))
                  else: pass
      mcmcresult['best-fit']['scaleamp'] = bfscaleamp
      
      bfshiftphase = np.zeros((len(data),2))
      if 'shiftphase' in mcmcresult.keys():
            for i,t in enumerate(mcmcresult['shiftphase']):
                  if i==0: pass # only matters if >1 datasets
                  elif t:
                        bfshiftphase[i][0] = np.median(c['astromshift_x_dset'+str(i)])
                        bfshiftphase[i][1] = np.median(c['astromshift_y_dset'+str(i)])
                        pbest.append(np.median(c['astromshift_x_dset'+str(i)]))
                        pbest.append(np.median(c['astromshift_y_dset'+str(i)]))
                  else: pass # no shifting
      mcmcresult['best-fit']['shiftphase'] = bfshiftphase
      
      mcmcresult['best-fit']['lnlike'] = calc_vis_lnlike(pbest,data,mcmcresult['best-fit']['lens'],
            mcmcresult['best-fit']['source'],
            Dd,Ds,Dds,ug,xmap,ymap,xemit,yemit,indices,
            sourcedatamap,scaleamp,shiftphase,modelcal)[0]
      
      # Calculate the deviance information criterion, using the Spiegelhalter+02 definition (cf Gelman+04)
      mcmcresult['best-fit']['DIC'] = -4*np.mean(mcmcresult['lnlike']) + 2*mcmcresult['best-fit']['lnlike']
      
      # If we did any modelcal stuff, keep the antenna phase offsets here
      if any(modelcal): 
            mcmcresult['modelcal'] = [True if j else False for j in modelcal]
            dp = np.squeeze(np.asarray([[a[1] for a in l if ~np.any(np.isnan(a[0]))] for l in blobs]))
            a = [x for l in dp for x in l] # Have to dick around with this if we had any nan's
            dphases = np.squeeze(np.reshape(a,(nwalkers*nstep-bad.sum(),len(data),-1),order='F'))
            if len(data) > 1: 
                  for i in range(len(data)):
                        if modelcal[i]: mcmcresult['calphases_dset'+str(i)] = np.vstack(dphases[:,i])
            else: 
                  if any(modelcal): mcmcresult['calphases_dset0'] = dphases
      
      return mcmcresult