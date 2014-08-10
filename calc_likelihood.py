import numpy as np
from PIL import Image
from scipy.fftpack import fftshift,fft2
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import convolve
from astropy.io import fits
import copy
from Model_objs import *
from RayTracePixels import *
from SourceProfile import SourceProfile
from Data_objs import Visdata
from modelcal import model_cal
arcsec2rad = np.pi/180/3600

__all__ = ['calc_vis_lnlike','calc_im_lnlike_galfit','pass_priors']

def calc_vis_lnlike(p,data,lens,source,shear,
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

      # First we'll take care of the priors implemented on the parameters.
      # If we pass them all, `pass_priors' returns updated versions of all the objects
      # Otherwise, we got False, and unpacking all those values raises a TypeError.
      x = pass_priors(p,lens,source,shear,scaleamp,shiftphase)
      try: thislens,thissource,thisshear,thisascale,thispshift = x
      except TypeError: return -np.inf,[np.nan]

      # Ok, if we've made it this far we can do the actual likelihood calculation
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
                        if thissource[jsrc].lensed: 
                              ims = SourceProfile(xsrc,ysrc,thissource[jsrc],thislens)
                              imsrc += ims
                              mags[jsrc] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./thissource[jsrc].flux['value']
                        else: immap += SourceProfile(xmap,ymap,thissource[jsrc],thislens); mags[jsrc] = 1.
            else: # Assume we fit all sources to this dataset
                  for j,src in enumerate(thissource):
                        if thissource.lensed: 
                              ims = SourceProfile(xsrc,ysrc,src,thislens)
                              imsrc += ims
                              mags[j] = ims.sum()*(xemit[0,1]-xemit[0,0])**2./src.flux['value']
                        else: immap += SourceProfile(xmap,ymap,thissource,thislens); mags[j] = 1.

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


def calc_im_lnlike_galfit(p,image,sigma,psf,lens,source,shear,
                    Dd,Ds,Dds):
      """
      Calculates log-likelihood of the given parameters.

      Inputs:
      p:
            A numpy array of arbitrary length, generated elsewhere to contain
            all the quantities we're fitting for. Has to be this way for emcee
            compatibility.
      image:
            An astropy.io PrimaryHDU object, modified to contain attributes
            related to lensing calculations by ImageModelMCMC
      sigma:
            An ndarray representing the uncertainty map of the above image.
      psf:
            An ndarray representing the PSF the above image has been convolved with
      lens:
            A lens object; will contain info about any fixed params, priors, etc.
      source:
            A source object or list of them.
      shear:
            A shear object (or None)

      Returns:
      lnL:
            The log-likelihood of the model given the data.
      mu:
            The magnification(s) of the lensed source(s)
      """

      # First we'll take care of the priors implemented on the parameters.
      # If we pass them all, `pass_priors' returns updated versions of all the objects
      # Otherwise, we got False, and unpacking all those values raises a TypeError.
      image = copy.deepcopy(image)
      x = pass_priors(p,lens,source,shear,[False],[False])
      try: thislens,thissource,thisshear,foo,bar = x
      except TypeError: return -np.inf,[np.nan]
      
      # Ok, if we've made it this far we can do the actual likelihood calculation
      
      # First, we'll set up the initial part of the galfit file, ie everything except source specification
      # --- TODO ---

      # Do the raytracing for this set of lens & shear params
      if isinstance(thislens,SIELens): xsrc,ysrc = RayTraceSIE(image.xemit,image.yemit,thislens,Dd,Ds,Dds,thisshear)

      # Now loop through all the datasets, fitting the requested sources to each. We'll also calculate
      # magnifications for each source, defined as the sum of the output flux / input flux
      # Thus, the returned magnification will be an array of length (# of unique sources)
      lnL,mags = 0., np.zeros(len(thissource))

            # Two arrays, one of the lensed sources, the other the full field and any unlensed sources
      imsrc = np.zeros(image.xemit.shape)
      immap = np.zeros(image.xmap.shape)

      for j,src in enumerate(thissource):
            if thissource.lensed: 
                  ims = SourceProfile(xsrc,ysrc,src,thislens)
                  imsrc += ims
                  mags[j] = ims.sum()*(image.xemit[0,1]-image.xemit[0,0])**2./src.flux['value']
            else: 
                  with open(galfitfile,'a') as f: # Here we'll append the info about this source to the GALFIT file
                        if isinstance(src,GaussSource): # a symmetric gaussian
                              f.write('0) gaussian\n')
                              # Calculate source position in pixels, galfit measures origin at image lower left(?)
                              xloc = (src.xoff['value']+thislens.x['value']-image.xmap.min()) / (image.xmap[0,1]-image.xmap[0,0])
                              yloc = (src.yoff['value']+thislens.y['value']-image.ymap.min()) / (image.ymap[1,0]-image.ymap[0,0])
                              varx = s
                              f.write('1) {0:.1f} {1:.1f} {2:.0f} {3:.0f}'.format(xloc,yloc,int(src.xoff['fixed']),int(src.yoff['fixed'])))
                              # Add in "total magnitude" - Jingzhe, I don't know what this means exactly...
                              f.write('2) {0:.3f} {1:.0f}'.format(src.flux['value'],int(src.flux['fixed'])))
                              # And the source FWHM
                              fwhmpix = (2*np.sqrt(2*np.log(2))*src.width['value'])/(image.xmap[0,1]-image.xmap[0,0])
                              f.write('3) {0:.3f} {1:.0f}'.format(fwhmpix,int(src.width['fixed']))
                              # the rest we can just do; it's a circular gaussian
                              # not sure what the Z) row is actually doing....
                              f.write('9) 1.0 0\n10) 0.0 0\nZ) 0')
                        elif isinstance(src,SersicSource): # A generic sersic profile, arbitrary PA
                              pass # TODO, blarg pixel coords are annoying
                  mags[j] = 1.
                        
                        

      # Try to reproduce matlab's antialiasing thing; this uses a 3lobe lanczos low-pass filter
      imsrc = Image.fromarray(imsrc)
      resize = np.array(imsrc.resize((int(image.indices[1]-image.indices[0]),int(image.indices[3]-image.indices[2])),Image.ANTIALIAS))
      immap[image.indices[2]:image.indices[3],image.indices[0]:image.indices[1]] += resize
      

      # Okay, now we have our lensed emission model, let's convolve it with the PSF
      # and subtract from the data before sending to galfit
      imconv = convolve(immap,psf,mode='constant')
      image.data -= imconv
      image.writeto('galfit_input.fits')
      
      # Now we run galfit on the difference image...
      os.system('galfit '+galfitfile)
      
      # At this point, we read in either the model image (and do the chi-2 calculation ourselves)
      # or the residual image, in which case it's just the sum of the resid/sigma**2. Below assumes we 
      # have the residual image... is that right?
      resid = fits.open('galfit_output.fits')[0]
      lnL = -(resid**2./sigma**2.).sum()

      # Last-ditch attempt to keep from hanging
      if np.isnan(lnL): lnL = -np.inf

      return lnL,[mags]
      
def pass_priors(p,lens,source,shear,scaleamp,shiftphase):
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
      shear
            An external shear object, or None.
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
      thisshear = copy.deepcopy(shear)
      thisascale = copy.deepcopy(scaleamp)
      thispshift = copy.deepcopy(shiftphase)
            
      ip = 0 # current index in p
      if isinstance(lens,SIELens):
            for key in ['x','y','M','e','PA']:
                  if not vars(lens)[key]['fixed']:
                        # A uniform prior
                        if p[ip] < vars(lens)[key]['prior'][0] or p[ip] > vars(lens)[key]['prior'][1]: return False
                        thislens.__dict__[key]['value'] = p[ip]
                        ip += 1
      # now do the source(s)
      for i,src in enumerate(source): # Source is a list of source objects
            if isinstance(src,GaussSource):
                  for key in ['xoff','yoff','flux','width']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return False
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1
            elif isinstance(src,SersicSource):
                  for key in ['xoff','yoff','flux','alpha','index','axisratio','PA']:
                        if not vars(src)[key]['fixed']:
                              if p[ip] < vars(src)[key]['prior'][0] or p[ip] > vars(src)[key]['prior'][1]: return False
                              thissource[i].__dict__[key]['value'] = p[ip]
                              ip += 1
      # now do shear, if any
      if shear is not None:
            for key in ['shear','shearangle']:
                  if not vars(shear)[key]['fixed']:
                        if p[ip] < vars(shear)[key]['prior'][0] or p[ip] > vars(shear)[key]['prior'][1]: return False
                        thisshear.__dict__[key]['value'] = p[ip]
                        ip += 1
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
            
      return thislens,thissource,thisshear,thisascale,thispshift
      