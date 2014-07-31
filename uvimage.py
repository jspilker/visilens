import numpy as np
from scipy.fftpack import fftshift,fft2
from Data_objs import Visdata
from uvgrid import uvgrid
from utils import box,expsinc
arcsec2rad = np.pi/180./3600.
c = 2.99792458e8 # in m/s

__all__ = ['uvimageslow']

def uvimageslow(visdata,imsize=256,pixsize=0.5):
      """
      Invert a set of visibilities to the image plane in the slowest way possible
      (ie, no gridding of visibilities, just straight-up summation)
      
      Inputs:
      visdata:
            Any visdata object
      imsize:
            Size of the output image in pixels
      pixsize:
            Pixel size, in arcseconds

      Returns:
      image:
            A 2D array containing the image of the inverted visibilities
      """

      x = np.linspace(-imsize*pixsize/2. * arcsec2rad,+imsize*pixsize/2. * arcsec2rad,imsize)
      x,y = np.meshgrid(x,x)

      im = np.array(np.zeros(x.shape),dtype=complex)

      for i in range(visdata.u.size):
            im += (visdata.real[i]+1j*visdata.imag[i])*np.exp(2*np.pi*1j*((visdata.u[i]*x)+(visdata.v[i]*y)))

      return im

def uvimage(visdata,imsize=256,pixsize=0.5,weighting='natural',convolution='expsinc'):
      """
      Faster imaging by gridding then FFTs.
      """
      
      # First grid up the data...
      binsize = (pixsize*imsize*arcsec2rad)**-1.
      gridded_data = uvgrid(visdata,gridsize=imsize,binsize=binsize,convolution=convolution)

            
      
