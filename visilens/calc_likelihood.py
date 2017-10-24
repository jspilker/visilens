import numpy as np
from PIL import Image
from scipy.fftpack import fftshift,fft2
from scipy.special import gamma, gammaincinv
from scipy.interpolate import RectBivariateSpline
import copy
from lensing import *
from class_utils import *
from utils import *

arcsec2rad = np.pi/180/3600
rad2arcsec = 3600.*180./np.pi
deg2rad = np.pi/180.

__all__ = ['calc_vis_lnlike','pass_priors','SourceProfile',
            'create_modelimage','fft_interpolate','model_cal','logp','logl']


def calc_vis_lnlike(p,data,lens,source,
                    Dd,Ds,Dds,ug,xmap,ymap,xemit,yemit,indices,
                    sourcedatamap=None,scaleamp=False,shiftphase=False,modelcal=True):
      """
      Calculates log-likelihood of the given parameters.

      Inputs:
      p:
            A numpy array of arbitrary length, generated elsewhere to contain
            all the quantities we're fitting for. Has to be this way for emcee
            compatibility.
      data:
            A visdata object or list of visdata objects containing the visibilities
            we're fitting for.
      lens:
            A lens object; will contain info about any fixed params, priors, etc.
      source:
            A source object or list of them.
      shear:
            A shear object (or None)
      sourcedatamap:
            Tells how the source(s) should be matched up with the dataset(s). See
            LensModelMCMC docstring for syntax.
      scaledata:
            Whether a relative rescaling between datasets should be allowed. See
            LensModelMCMC docstring for syntax.
      modelcal:
            Whether to perform the pseudo-selfcal of Hezaveh+13 to compensate for
            uncorrected phase errors.

      Returns:
      lnL:
            The log-likelihood of the model given the data.
      """
      
      # First we'll take care of the priors implemented on the parameters.
      # If we pass them all, `pass_priors' returns updated versions of all the objects
      # Otherwise, we got False, and unpacking all those values raises a TypeError.
      x = pass_priors(p,lens,source,scaleamp,shiftphase)
      try: thislens,thissource,thisascale,thispshift = x
      except TypeError: return -np.inf,[np.nan]
      
      # Ok, if we've made it this far we can do the actual likelihood calculation
      # Loop through all the datasets, fitting the requested sources to each. We'll also calculate
      # magnifications for each source, defined as the sum of the output flux / input flux
      # Thus, the returned magnification will be an array of length (# of sources)
      lnL,dphases = 0., [[]]*len(data)
      mus = np.zeros(len(source))
      for i,dset in enumerate(data):
      
            # Make a model of this field.
            immap,mags = create_modelimage(thislens,thissource,\
                  xmap,ymap,xemit,yemit,indices,Dd,Ds,Dds,sourcedatamap[i])
                  
            # Filter non-zero magnifications (only relevant if sourcedatamap)
            mus[mags != 0 ] = mags[mags != 0]
            
            # ... and interpolate/sample it at our uv coordinates
            interpdata = fft_interpolate(dset,immap,xmap,ymap,ug,thisascale[i],thispshift[i])
            
            # If desired, do model-cal on this dataset
            if modelcal[i]:
                  modeldata,dphase = model_cal(dset,interpdata,modelcal[i][0],modelcal[i][1])
                  dphases[i] = dphase
                  lnL -= (((modeldata.real - interpdata.real)**2. + (modeldata.imag - interpdata.imag)**2.)/modeldata.sigma**2.).sum()
                  
            # Calculate the contribution to chi2 from this dataset
            else: lnL -= (((dset.real - interpdata.real)**2. + (dset.imag - interpdata.imag)**2.)/dset.sigma**2.).sum()

      # Last-ditch attempt to keep from hanging
      if np.isnan(lnL): return -np.inf,[np.nan]
      
      return lnL,[mus,dphases]
      
def pass_priors(p,lens,source,scaleamp,shiftphase):
      """
      Figures out if any of the proposed values in `p' exceed the priors
      defined by each object.
      
      Parameters:
      p
            An array of the emcee-proposed steps for each of the free
            parameters we're dealing with here
      lens
            Any of the currently implemented lens objects
      source
            Any one or a list of sources
      scaleamp
            Either a single True/False value, list of these, or None.
      shiftphase
            Same as above
      
      Returns:
      False if we failed a prior, otherwise returns copies of 
      all the passed objects with the free param values set to the values
      that exist in `p'
      """
      # Our objects are mutable; for the versions updated with the values
      # in `p', we muse make copies to avoid effing things up.
      thislens = copy.deepcopy(lens)
      thissource = copy.deepcopy(source)
      thisascale = copy.deepcopy(scaleamp)
      thispshift = copy.deepcopy(shiftphase)
            
      ip = 0 # current index in p
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__=='SIELens':
                  for key in ['x','y','M','e','PA']:
                        if not vars(ilens)[key]['fixed']:
                              # A uniform prior
                              if p[ip] < vars(ilens)[key]['prior'][0] or p[ip] > vars(ilens)[key]['prior'][1]: return False
                              thislens[i]._altered = True
                              thislens[i].__dict__[key]['value'] = p[ip]
                              ip += 1
            elif ilens.__class__.__name__=='ExternalShear':
                  for key in ['shear','shearangle']:
                        if not vars(ilens)[key]['fixed']:
                              if p[ip] < vars(ilens)[key]['prior'][0] or p[ip] > vars(ilens)[key]['prior'][1]: return False
                              thislens[i]._altered = True
                              thislens[i].__dict__[key]['value'] = p[ip]
                              ip += 1
      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if src.__class__.__name__=='GaussSource':
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return False
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1
            elif src.__class__.__name__=='SersicSource':
                  for key in ['xoff','yoff','flux','majax','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return False
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1
            elif src.__class__.__name__=='PointSource':
                  for key in ['xoff','yoff','flux']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return False
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1                              
      
      # Amplitude re-scaling for multiple datasets
      for i,t in enumerate(scaleamp): # only matters if >1 datasets
            if i==0: thisascale[i] = 1.
            elif t:
                  if p[ip] < 0.5 or p[ip] > 2.: return False
                  thisascale[i] = p[ip]
                  ip += 1
            else: thisascale[i] = 1.
      # Last do phase/astrometric shifts, can only be +/- 0.5" in each direction (reasonable?)
      for i,t in enumerate(shiftphase):
            if i==0: thispshift[i] = [0.,0.] # only matters if >1 datasets
            elif t:
                  if p[ip] < -0.5 or p[ip] > 0.5: return False
                  if p[ip+1] < -0.5 or p[ip+1] > 0.5: return False
                  thispshift[i] = [p[ip], p[ip+1]]
                  ip += 2
            else: thispshift[i] = [0.,0.] # no shifting
            
      return thislens,thissource,thisascale,thispshift


def SourceProfile(xsource,ysource,source,lens):
      """
      Creates the source-plane profile of the given Source.

      Inputs:
      xsource,ysource:
            Source-plane coordinates, in arcsec, on which to
            calculate the luminosity profile of the source
      
      Source:
            Any supported source-plane object, e.g. a GaussSource
            object. The object will contain all the necessary
            parameters to create the profile.

      Lens:
            Any supported Lens object, e.g. an SIELens. We only need
            this because, in the case of single lenses, the source
            position is defined as offset from the lens centroid. If
            there is more than one lens, or if the source is unlensed,
            the source position is defined **relative to the field 
            center, aka (0,0) coordinates**.
            

      Returns:
      I:
            The luminosity profile of the given Source. Has same
            shape as xsource and ysource. Note: returned image has
            units of flux / arcsec^2 (or whatever the x,y units are),
            so to properly normalize, must multiply by pixel area. This
            isn't done here since the lensing means the pixels likely
            aren't on a uniform grid.
      """
      
      lens = list(np.array([lens]).flatten())

      # First case: a circular Gaussian source.
      if source.__class__.__name__=='GaussSource':
            sigma = source.width['value']
            amp   = source.flux['value']/(2.*np.pi*sigma**2.)
            if source.lensed:# and len(lens)==1:
                  xs = source.xoff['value'] + lens[0].x['value']
                  ys = source.yoff['value'] + lens[0].y['value']
            else:
                  xs = source.xoff['value']
                  ys = source.yoff['value']
            
            return amp * np.exp(-0.5 * (np.sqrt((xsource-xs)**2.+(ysource-ys)**2.)/sigma)**2.)

      elif source.__class__.__name__=='SersicSource':
            if source.lensed:# and len(lens)==1:
                  xs = source.xoff['value'] + lens[0].x['value']
                  ys = source.yoff['value'] + lens[0].y['value']
            else:
                  xs = source.xoff['value']
                  ys = source.yoff['value']
            PA, ar = source.PA['value']*deg2rad, source.axisratio['value']
            majax, index = source.majax['value'], source.index['value']
            dX = (xsource-xs)*np.cos(PA) + (ysource-ys)*np.sin(PA)
            dY = (-(xsource-xs)*np.sin(PA) + (ysource-ys)*np.cos(PA))/ar
            R = np.sqrt(dX**2. + dY**2.)
            
            # Calculate b_n, to make reff enclose half the light; this approx from Ciotti&Bertin99
            # This approximation good to 1 in 10^4 for n > 0.36; for smaller n it gets worse rapidly!!
            #bn = 2*index - 1./3. + 4./(405*index) + 46./(25515*index**2) + 131./(1148175*index**3) - 2194697./(30690717750*index**4)
            # Note, now just calculating directly because everyone's scipy
            # should be sufficiently modern.
            bn = gammaincinv(2. * index, 0.5)
            
            # Backing out from the integral to R=inf of a general sersic profile
            Ieff = source.flux['value'] * bn**(2*index) / (2*np.pi*majax**2 * ar * np.exp(bn) * index * gamma(2*index))
            
            return Ieff * np.exp(-bn*((R/majax)**(1./index)-1.))
      
      elif source.__class__.__name__=='PointSource':
            if source.lensed:# and len(lens)==1:
                  xs = source.xoff['value'] + lens[0].x['value']
                  ys = source.yoff['value'] + lens[0].y['value']
                  return ValueError("Lensed point sources not working yet... try a"\
                   "gaussian with small width instead...")
            else:
                  xs = source.xoff['value']
                  ys = source.yoff['value']
                  
            yloc = np.abs(xsource[0,:] - xs).argmin()
            xloc = np.abs(ysource[:,0] - ys).argmin()
            
            m = np.zeros(xsource.shape)
            m[xloc,yloc] += source.flux['value']/(xsource[0,1]-xsource[0,0])**2.
            
            return m
            
      
      else: raise ValueError("So far only GaussSource, SersicSource, and "\
            "PointSource objects supported...")

def create_modelimage(lens,source,xmap,ymap,xemit,yemit,indices,
      Dd=None,Ds=None,Dds=None,sourcedatamap=None):
      """
      Creates a model lensed image given the objects and map
      coordinates specified.  Supposed to be common for both
      image fitting and visibility fitting, so we don't need
      any data here.

      Returns:
      immap
            A 2D array representing the field evaluated at
            xmap,ymap with all sources included.
      mus:
            A numpy array of length N_sources containing the
            magnifications of each source (1 if unlensed).
      """
      
      lens = list(np.array([lens]).flatten()) # Ensure lens(es) are a list
      source = list(np.array([source]).flatten()) # Ensure source(s) are a list
      mus = np.zeros(len(source))
      immap, imsrc = np.zeros(xmap.shape), np.zeros(xemit.shape)

      # If we didn't get pre-calculated distances, figure them here assuming Planck15
      if np.any((Dd is None,Ds is None, Dds is None)):
            from astropy.cosmology import Planck15 as cosmo
            Dd = cosmo.angular_diameter_distance(lens[0].z).value
            Ds = cosmo.angular_diameter_distance(source[0].z).value
            Dds= cosmo.angular_diameter_distance_z1z2(lens[0].z,source[0].z).value

      # Do the raytracing for this set of lens & shear params
      xsrc,ysrc = LensRayTrace(xemit,yemit,lens,Dd,Ds,Dds)

      if sourcedatamap is not None: # ... then particular source(s) are specified for this map
            for jsrc in sourcedatamap:
                  if source[jsrc].lensed: 
                        ims = SourceProfile(xsrc,ysrc,source[jsrc],lens)
                        imsrc += ims
                        mus[jsrc] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./source[jsrc].flux['value']
                  else: immap += SourceProfile(xmap,ymap,source[jsrc],lens); mus[jsrc] = 1.
      else: # Assume we put all sources in this map/field
            for j,src in enumerate(source):
                  if src.lensed: 
                        ims = SourceProfile(xsrc,ysrc,src,lens)
                        imsrc += ims
                        mus[j] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./src.flux['value']
                  else: immap += SourceProfile(xmap,ymap,src,lens); mus[j] = 1.

      # Try to reproduce matlab's antialiasing thing; this uses a 3lobe lanczos low-pass filter
      imsrc = Image.fromarray(imsrc)
      resize = np.array(imsrc.resize((int(indices[1]-indices[0]),int(indices[3]-indices[2])),Image.ANTIALIAS))
      immap[indices[2]:indices[3],indices[0]:indices[1]] += resize

      # Flip image to match sky coords (so coordinate convention is +y = N, +x = W, angle is deg E of N)
      immap = immap[::-1,:]

      # last, correct for pixel size.
      immap *= (xmap[0,1]-xmap[0,0])**2.

      return immap,mus


def fft_interpolate(visdata,immap,xmap,ymap,ug=None,scaleamp=1.,shiftphase=[0.,0.]):
      """
      Take a dataset and a map of a field, fft the image,
      and interpolate onto the uv-coordinates of the dataset.

      Returns:
      interpdata: Visdata object
            A dataset which samples the given immap
      """
      
      # Correct for PB attenuation            
      if visdata.PBfwhm is not None: 
            PBs = visdata.PBfwhm / (2.*np.sqrt(2.*np.log(2)))
            immap *= np.exp(-(xmap**2./(2.*PBs**2.)) - (ymap**2./(2.*PBs**2.)))
      
      #immap = immap[::-1,:] # Fixes issue of origin in tlc vs blc to match sky coords  
      imfft = fftshift(fft2(fftshift(immap)))
      
      # Calculate the uv points we need, if we don't already have them
      if ug is None:
            kmax = 0.5/((xmap[0,1]-xmap[0,0])*arcsec2rad)
            ug = np.linspace(-kmax,kmax,xmap.shape[0])

      # Interpolate the FFT'd image onto the data's uv points
      # Using RBS, much faster since ug is gridded
      spliner = RectBivariateSpline(ug,ug,imfft.real,kx=1,ky=1)
      splinei = RectBivariateSpline(ug,ug,imfft.imag,kx=1,ky=1)
      interpr = spliner.ev(visdata.v,visdata.u)
      interpi = splinei.ev(visdata.v,visdata.u)
      interpdata = Visdata(visdata.u,visdata.v,interpr,interpi,visdata.sigma,\
            visdata.ant1,visdata.ant2,visdata.PBfwhm,'interpolated_data')

      # Apply scaling, phase shifts; wrap phases to +/- pi.
      interpdata.amp *= scaleamp
      interpdata.phase += 2.*np.pi*arcsec2rad*(shiftphase[0]*interpdata.u + shiftphase[1]*interpdata.v)
      interpdata.phase = (interpdata.phase + np.pi) % (2*np.pi) - np.pi

      return interpdata


def model_cal(realdata,modeldata,dPhi_dphi=None,FdPC=None):
      """
      Routine following Hezaveh+13 to implement perturbative phase corrections
      to model visibility data. This routine is designed to start out from an
      intermediate step in the self-cal process, since we can avoid doing a
      lot of expensive matrix inversions that way, but if either of the
      necessary arrays aren't provided, we calculate them.

      Inputs:
      realdata,modeldata:
            visdata objects containing the actual data and model-generated data

      dPhi_dphi: None, or pre-calculated
            See H+13, App A. An N_ant-1 x M_vis matrix whose ik'th element is 1
            if the first antenna of the visibility is k, -1 if the second, or 0.

      FdPC: None, or pre-calculated
            See H+13, App A, eq A2. This is an N_ant-1 x M_vis matrix, equal to
            -inv(F)*dPhi_dphi*inv(C) in the nomenclature of H+13. This has the 
            matrix inversions that we want to avoid calculating at every MCMC
            iteration (inverting C takes ~3s for M~5k, even with a sparse matrix).

      Outputs:
      modelcaldata:
            visdata object containing updated visibilities

      dphi:
            Array of length N_ant-1 containing the implemented phase offsets
      """

      # If we don't have the pre-calculated arrays, do so now. It's expensive
      # to do these matrix inversions at every MCMC step.
      if np.any((FdPC is None, dPhi_dphi is None)):
            import scipy.sparse
            uniqant = np.unique(np.asarray([realdata.ant1,realdata.ant2]).flatten())
            dPhi_dphi = np.zeros((uniqant.size-1,realdata.u.size))
            for j in range(1,uniqant.size):
                  dPhi_dphi[j-1,:] = (realdata.ant1==uniqant[j])-1*(realdata.ant2==uniqant[j])
            C = scipy.sparse.diags((realdata.sigma/realdata.amp)**-2.,0)
            F = np.dot(dPhi_dphi,C*dPhi_dphi.T)
            Finv = np.linalg.inv(F)
            FdPC = np.dot(-Finv,dPhi_dphi*C)

      # Calculate current phase difference between data and model; wrap to +/- pi
      deltaphi = realdata.phase - modeldata.phase
      deltaphi = (deltaphi + np.pi) % (2 * np.pi) - np.pi

      dphi = np.dot(FdPC,deltaphi)

      modelcaldata = copy.deepcopy(realdata)
      
      modelcaldata.phase += np.dot(dPhi_dphi.T,dphi)

      return modelcaldata,dphi

def logp(p): return 0.0 # flat prior, this is handled in calc_likelihood

def logl(p,**kwargs): return calc_vis_lnlike(p,**kwargs)[0] # PT sampler can't deal with blobs

