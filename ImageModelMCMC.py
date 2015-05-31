import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import emcee
from Model_objs import *
from GenerateLensingGrid import GenerateLensingGrid
from calc_likelihood import calc_im_lnlike_galfit

arcsec2rad = np.pi/180/3600

def ImageModelMCMC(image,psf,sigma,lens,source,shear=None,
                  highresbox=[-3.,3.,-3.,3.],emitres=None,
                  #GALFITPARAMS GO HERE....
                  cosmo=None,nwalkers=1e3,nburn=1e3,nstep=1e3,nthreads=1):
      """
      Wrapper function which basically takes what the user wants and turns it into the
      format needed for the acutal MCMC lens modeling.
      
      Inputs:
      image:
            An astropy.io PrimaryHDU object. Since we're mediating through galfit,
            it's easiest just to work directly with these. We calculate the lensed
            emission based on the header keywords CDELT_1&2 or CD1_1&2 and the field
            size.
      sigma:
            An astropy.io PrimaryHDU object representing the uncertainty map
            for the above image.
      psf:
            An astropy.io PrimaryHDU object representing the PSF through which the
            input image above was convolved. Should have the same pixel scale as 
            the input image.
      lens:
            Any of the currently implemented lens objects.
      source:
            One or more of the currently implemented source objects; if more than
            one source to be fit, should be a list of multiple sources.
      shear:
            An ExternalShear object, or None (default) if no external shear desired
      highresbox:
            A list of [xmin,xmax,ymin,ymax] which tells the region to perform the lensing
            calculations at high resolution, to account for large magnification gradients.
            Just has to conservatively encompass the lensed emission.
      emitres:
            Resolution of the above highresbox; if None, a suitable value is picked.
      nwalkers:
            Number of walkers to use in the mcmc process; see dan.iel.fm/emcee/current
            for more details.
      nburn:
            Number of burn-in steps to take with the chain.
      nstep:
            Number of actual steps to take in the mcmc chains after the burn-in
      nthreads:
            Number of threads (read: cores) to use during the fitting, default 1.

      Returns:
      mcmcresult:
            A nested dict containing the chains requested. Will have all the MCMC
            chain results, plus metadata about the run (initial params, data used,
            etc.). Formatting still a work in progress.
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

      # Making these lists just makes later stuff easier since we now know the dtype
      source = list(np.array([source]).flatten()) # Ensure source(s) are a list
      data = list(np.array([data]).flatten())     # Same for dataset(s)

      # emcee isn't very flexible in terms of how it gets initialized; start by
      # assembling the user-provided info into a form it likes
      ndim, p0, colnames = 0, [], []
      # Lens first
      if isinstance(lens,SIELens):
            for key in ['x','y','M','e','PA']:
                  if not vars(lens)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(lens)[key]['value'])
                        colnames.append(key+'L')
      # Then source(s)
      for i,src in enumerate(source):
            if isinstance(src,GaussSource):
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(src)[key]['value'])
                              colnames.append(key+'S'+str(i))
            elif isinstance(src,SersicSource):
                  for key in ['xoff','yoff','flux','alpha','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              ndim += 1
                              p0.append(vars(src)[key]['value'])
                              colnames.append(key+'S'+str(i))
      # Then shear
      if shear is not None:
            for key in ['shear','shearangle']:
                  if not vars(shear)[key]['fixed']:
                        ndim += 1
                        p0.append(vars(shear)[key]['value'])
                        colnames.append(key)


      # Figure out the x&y coordinates of each pixel in the image and add
      # them as attributes of the image object.
      dx,dy = None, None
      if 'CDELT1' in image.header: dx = image.header['CDELT1']*3600 # to arcsec
      if 'CDELT2' in image.header: dy = image.header['CDELT2']*3600
      if 'CD1_1' in image.header and 'CD1_2' in image.header: 
            dx = 3600*np.sqrt(image.header['CD1_1']**2. + image.header['CD1_2']**2.)
      if 'CD2_1' in image.header and 'CD2_2' in image.header:
            dy = 3600*np.sqrt(image.header['CD2_1']**2. + image.header['CD2_2']**2.)
      if dx is None or dy is None: 
            raise KeyError("I don't know the pixel scale of your image. Does your fits"\
                  "header have CDELT1&2 or CD[12]_[12] in it?")
      xm = np.arange(-dx*image.shape[0]/2.,+dx*image.shape[0]/2.,dx)
      ym = np.arange(-dy*image.shape[1]/2.,+dy*image.shape[1]/2.,dy)
      image.x,image.y = np.meshgrid(xm,ym)

      # Create our lensing grid coordinates now, since those shouldn't be
      # recalculated with every call to the likelihood function
      if emitres is None: emitres = np.min((image.x[0,1]-image.x[0,0],image.y[1,0]-image.y[0,0]))/4.
      ixx = np.round(np.interp(np.asarray(highresbox)[:2],xm,np.arange(image.shape[0])))
      ixy = np.round(np.interp(np.asarray(highresbox)[2:],ym,np.arange(image.shape[1])))
      indices = np.concatenate((ixx,ixy))
      Nemx = 1 + np.abs(ixx[1]-ixx[0])*np.ceil(dx/emitres)
      Nemy = 1 + np.abs(ixy[1]-ixy[0])*np.ceil(dy/emitres)
      xemcoords = np.linspace(image.x[ixx[1],ixx[0]],image.x[ixx[0],ixx[1]],Nemx)
      yemcoords = np.linspace(image.y[ixy[0],ixy[1]],image.y[ixy[1],ixy[0]],Nemy)
      xemit,yemit = np.meshgrid(xemcoords,yemcoords)
      image.xemit, image.yemit = xemit, yemit
      image.indices = indices

      # Calculate some distances; we only need to calculate these once.
      # This assumes multiple sources are all at same z; should be this
      # way anyway or else we'd have to deal with multiple lensing planes
      if cosmo is None: from astropy.cosmology import WMAP9 as cosmo
      Dd = cosmo.angular_diameter_distance(lens.z).value
      Ds = cosmo.angular_diameter_distance(source[0].z).value
      Dds= cosmo.angular_diameter_distance_z1z2(lens.z,source[0].z).value

      p0 = np.array(p0)
      # Create a ball of starting points for the walkers, gaussian ball of 
      # 10% width; if initial value is 0 (eg, astrometric shift), give a small sigma
      initials = emcee.utils.sample_ball(p0,np.asarray([0.1*x if x else 0.05 for x in p0]),int(nwalkers))

      # Create the sampler object; uses calc_likelihood function defined elsewhere
      lenssampler = emcee.EnsembleSampler(nwalkers,ndim,calc_im_lnlike_galfit,
            args = [image,sigma.data,psf.data,lens,source,shear,Dd,Ds,Dds],
                    #TODO: GALFIT PARAMS HERE?],
            threads=nthreads)
      
      # Run burn-in phase
      print "Running burn-in... "
      pos,prob,rstate,mus = lenssampler.run_mcmc(initials,nburn,storechain=False)
      lenssampler.reset()
      
      # Run actual chains
      print "Done. Running chains... "
      lenssampler.run_mcmc(pos,nstep,rstate0=rstate)
      print "Mean acceptance fraction: ",np.mean(lenssampler.acceptance_fraction)

      blobs = lenssampler.blobs
      #mus = np.asarray([[a[0] for a in l] for l in blobs]).flatten()
      #colnames.append('mu')

      # Assemble the output. Want to return something that contains both the MCMC chains
      # themselves, but also metadata about the run.
      mcmcresult = {}

      try: # keep track of mercurial revision, for reproducibility's sake
            import subprocess
            mcmcresult['hghash'] = subprocess.check_output('hg id -i',shell=True).rstrip()
      except:
            mcmcresult['hghash'] = 'None'

      mcmcresult['datasets'] = [dset.filename for dset in data] # Data files used

      mcmcresult['lens_p0'] = lens      # Initial params for lens,src(s),shear; also tells if fixed, priors, etc.
      mcmcresult['source_p0'] = source
      if shear: mcmcresult['shear_p0'] = shear

      if sourcedatamap: mcmcresult['sourcedatamap'] = sourcedatamap

      #mcmcresult['chains'] = np.core.records.fromarrays(np.c_[lenssampler.flatchain,mus].T,names=colnames)
      mcmcresult['chains'] = np.core.records.fromarrays(np.c_[lenssampler.flatchain].T,names=colnames)
      mcmcresult['lnlike'] = lenssampler.flatlnprobability


      #if any(modelcal): mcmcresult['modelcal'] = {}
      #for i in range(len(data)):
      #      if modelcal[i]: mcmcresult['modelcal']['phases_dset'+str(i)] = np.vstack(dphases[:,i])

      return mcmcresult,lenssampler.flatchain,lenssampler.blobs,colnames
      #return lenssampler.flatchain,lenssampler.blobs,colnames

            
