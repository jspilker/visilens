import numpy as np
from scipy.fftpack import fftshift,fft2
from Data_objs import Visdata
import copy
from uvgrid import uvgrid
from utils import box,expsinc
arcsec2rad = np.pi/180./3600.
c = 2.99792458e8 # in m/s

__all__ = ['uvimageslow']

def uvimageslow(visdata,imsize=256,pixsize=0.5,taper=0.):
      """
      Invert a set of visibilities to the image plane in the slowest way possible
      (ie, no gridding of visibilities, just straight-up summation). This gets
      pretty darn close to imaging with CASA with natural weighting, no cleaning.
      
      Inputs:
      visdata:
            Any Visdata object
      imsize:
            Size of the output image in pixels
      pixsize:
            Pixel size, in arcseconds
      taper:
            Apply an additional gaussian taper to the
            visibilities with sigma of this, in arcsec.

      Returns:
      image:
            A 2D array containing the image of the inverted visibilities
      """

      thisvis = copy.deepcopy(visdata)

      if taper > 0.:
            uvsig = (taper * arcsec2rad)**-1.
            scalefac = np.exp(-thisvis.uvdist**2/(2*uvsig**2.))
            thisvis.sigma /= scalefac

      x = np.linspace(-imsize*pixsize/2. * arcsec2rad,+imsize*pixsize/2. * arcsec2rad,imsize)
      x,y = np.meshgrid(x,x)
      # These offsets line us up with CASA's coord system
      x -= pixsize/2. * arcsec2rad
      y += pixsize/2. * arcsec2rad

      im = np.array(np.zeros(x.shape),dtype=complex)

      # Check to see if we have the conjugate visibilities, and image.
      if np.all(thisvis.u[:thisvis.u.size/2] == -thisvis.u[thisvis.u.size/2:]):
            for i in range(thisvis.u.size/2):
                  im += (thisvis.sigma[i]**-2)*(thisvis.real[i]+1j*thisvis.imag[i]) *\
                        np.exp(2*np.pi*1j*((thisvis.u[i]*x)+(thisvis.v[i]*y)))
      else:
            for i in range(thisvis.u.size):
                  im += (thisvis.sigma[i]**-2)*(thisvis.real[i]+1j*thisvis.imag[i]) *\
                        np.exp(2*np.pi*1j*((thisvis.u[i]*x)+(thisvis.v[i]*y)))
            
      # We only imaged the non-conjugate visibilities; fix and renormalize to Jy/beam units
      im = 2*im.real
      im /= (2*thisvis.sigma**-2).sum()

      return im

def uvimage(visdata,imsize=256,pixsize=0.5,weighting='natural',convolution='expsinc'):
      """
      Faster imaging by gridding then FFTs.
      """
      
      # First grid up the data...
      binsize = (pixsize*imsize*arcsec2rad)**-1.
      gridded_data = uvgrid(visdata,gridsize=imsize,binsize=binsize,convolution=convolution)

            
      
