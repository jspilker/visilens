import numpy as np
from scipy.special import gamma
from Model_objs import *
arcsec2rad = (np.pi/(180.*3600.))
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.

def SourceProfile(xsource,ysource,Source,Lens):
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
            Any supported Lens object, e.g. an SIELens.
            Just needed because the Source x&y center positions are
            defined relative to the lens centroid.

      Returns:
      I:
            The luminosity profile of the given Source. Has same
            shape as xsource and ysource. Note: returned image has
            units of flux / arcsec^2 (or whatever the x,y units are),
            so to properly normalize, must multiply by pixel area. This
            isn't done here since the lensing means the pixels likely
            aren't on a uniform grid.
      """

      # First case: a circular Gaussian source.
      if isinstance(Source,GaussSource):
            sigma = Source.width['value']
            amp   = Source.flux['value']/(2.*np.pi*sigma**2.)
            xs = Source.xoff['value'] + Lens.x['value']
            ys = Source.yoff['value'] + Lens.y['value']
            
            return amp * np.exp(-0.5 * (np.sqrt((xsource-xs)**2.+(ysource-ys)**2.)/sigma)**2.)

      elif isinstance(Source,SersicSource):
            xs = Source.xoff['value'] + Lens.x['value']
            ys = Source.yoff['value'] + Lens.y['value']
            PA, ar = Source.PA['value']*deg2rad, Source.axisratio['value']
            alpha, index = Source.alpha['value'], Source.index['value']
            dX = (xsource-xs)*np.cos(PA) + (ysource-ys)*np.sin(PA)
            dY = (-(xsource-xs)*np.sin(PA) + (ysource-ys)*np.cos(PA*deg2rad))/ar
            R = np.sqrt(dX**2. + dY**2.)
            # Backing out from the integral to R=inf of a general sersic profile
            I0 = Source.flux['value']/(2*np.pi * alpha**2. * ar * index * gamma(2*index))
            
            return I0 * np.exp(-(R/alpha)**(1./index))
      
      else: raise ValueError("So far only GaussSource and SersicSource objects supported...")
