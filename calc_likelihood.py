import numpy as np
import Image
from scipy.fftpack import fftshift,fft2
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import zoom,gaussian_filter
import copy
from Model_objs import *
from RayTracePixels import *
from SourceProfile import SourceProfile
from Data_objs import Visdata
from modelcal import model_cal
arcsec2rad = np.pi/180/3600

def calc_likelihood(p,data,lens,source,shear,
                    Dd,Ds,Dds,kmax,xmap,ymap,xemit,yemit,indices,
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

      # First we'll take care of the priors implemented on the parameters; lens
      # params are the first given in 'p'. Also set up current versions of the
      # objects at the current parameters.
      thislens = copy.deepcopy(lens)
      thissource = copy.deepcopy(source)
      thisshear = copy.deepcopy(shear)
      thisascale = copy.deepcopy(scaleamp)
      thispshift = copy.deepcopy(shiftphase)
      ip = 0 # current index in p
      if isinstance(lens,SIELens):
            for key in ['x','y','M','e','PA']:
                  if not vars(lens)[key]['fixed']:
                        # A uniform prior
                        if p[ip] < vars(lens)[key]['prior'][0] or p[ip] > vars(lens)[key]['prior'][1]: return -np.inf,[np.nan]
                        thislens.__dict__[key]['value'] = p[ip]
                        ip += 1
      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if isinstance(src,GaussSource):
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return -np.inf,[np.nan]
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1
      # now do shear, if any
      if shear is not None:
            for key in ['shear','shearangle']:
                  if not vars(shear)[key]['fixed']:
                        if p[ip] < vars(shear)[key]['prior'][0] or p[ip] > vars(shear)[key]['prior'][1]: return -np.inf,[np.nan]
                        thisshear.__dict__[key]['value'] = p[ip]
                        ip += 1
      for i,t in enumerate(scaleamp): # only matters if >1 datasets
            if i==0: thisascale[i] = 1.
            elif t:
                  if p[ip] < 0.5 or p[ip] > 2.: return -np.inf,[np.nan]
                  thisascale[i] = p[ip]
                  ip += 1
            else: thisascale[i] = 1.
      # Last do phase/astrometric shifts, can only be +/- 0.5" in each direction (reasonable?)
      for i,t in enumerate(shiftphase):
            if i==0: thispshift[i] = [0.,0.] # only matters if >1 datasets
            elif t:
                  if p[ip] < -0.5 or p[ip] > 0.5: return -np.inf,[np.nan]
                  if p[ip+1] < -0.5 or p[ip+1] > 0.5: return -np.inf,[np.nan]
                  thispshift[i] = [p[ip], p[ip+1]]
                  ip += 2
            else: thispshift[i] = [0.,0.] # no shifting

      # Ok, if we've made it this far, then we pass all the priors, so we can do the actual likelihood calculation
      ug = np.linspace(-kmax,kmax,xmap.shape[0])

      # Do the raytracing for this set of lens & shear params
      if isinstance(thislens,SIELens): xsrc,ysrc = RayTraceSIE(xemit,yemit,thislens,Dd,Ds,Dds,thisshear)

      # Now loop through all the datasets, fitting the requested sources to each. We'll also calculate
      # magnifications for each source, defined as the sum of the output flux / input flux
      # Thus, the returned magnification will be an array of length (# of unique sources)
      lnL,mags,dphases = 0., np.zeros(len(thissource)),[[]]*len(data)
      for i,dset in enumerate(data):

            # Two arrays, one of the lensed sources, the other the full field and any unlensed sources
            imsrc = np.zeros(xemit.shape)
            immap = np.zeros(xmap.shape)

            if sourcedatamap is not None: # ... then particular source(s) are specified for each dataset
                  for jsrc in sourcedatamap[i]:
                        ims = SourceProfile(xsrc,ysrc,thissource[jsrc],thislens)
                        imsrc += ims
                        mags[jsrc] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./thissource[jsrc].flux['value']
            else: # Assume we fit all sources to this dataset
                  for j,src in enumerate(thissource):
                        ims = SourceProfile(xsrc,ysrc,src,thislens)
                        imsrc += ims
                        mags[j] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./src.flux['value']

            # Try to reproduce matlab's antialiasing thing; this uses a 3lobe lanczos low-pass filter
            imsrc = Image.fromarray(imsrc)
            resize = np.array(imsrc.resize((int(indices[1]-indices[0]),int(indices[3]-indices[2])),Image.ANTIALIAS))
            immap[indices[2]:indices[3],indices[0]:indices[1]] += resize

            # Correct for PB attenuation            
            if dset.PBfwhm is not None: 
                  PBs = dset.PBfwhm / (2.*np.sqrt(2.*np.log(2)))
                  immap *= np.exp(-(xmap**2./(2.*PBs**2.)) - (ymap**2./(2.*PBs**2.)))
            
            #immap = immap[::-1,:] # Fixes issue of origin in tlc vs blc to match sky coords  
            imfft = fftshift(fft2(fftshift(immap*(xmap[0,1]-xmap[0,0])**2.)))
            
            # Interpolate the FFT'd image onto the data's uv points
            # Using RBS, much faster since ug is gridded
            spliner = RectBivariateSpline(ug,ug,imfft.real,kx=1,ky=1)
            splinei = RectBivariateSpline(ug,ug,imfft.imag,kx=1,ky=1)
            interpr = spliner.ev(dset.v,dset.u)
            interpi = splinei.ev(dset.v,dset.u)
            interpdata = Visdata(dset.u,dset.v,interpr,interpi,np.zeros(dset.u.size))

            # Apply scaling, phase shifts; wrap phases to +/- pi.
            interpdata.amp *= thisascale[i]
            interpdata.phase += 2.*np.pi*arcsec2rad*(thispshift[i][0]*interpdata.u + thispshift[i][1]*interpdata.v)
            interpdata.phase = (interpdata.phase + np.pi) % (2*np.pi) - np.pi

            # If desired, do model-cal on this dataset
            if modelcal[i]:
                  modeldata,dphase = model_cal(dset,interpdata,modelcal[i][0],modelcal[i][1])
                  dphases[i] = dphase
                  lnL -= (((modeldata.real - interpdata.real)**2. + (modeldata.imag - interpdata.imag)**2.)/modeldata.sigma**2.).sum()

            # Calculate the contribution to chi2 from this dataset
            else: lnL -= (((dset.real - interpdata.real)**2. + (dset.imag - interpdata.imag)**2.)/dset.sigma**2.).sum()

      # Last-ditch attempt to keep from hanging
      if np.isnan(lnL): lnL = -np.inf

      return lnL,[mags,dphases]
