import numpy as np
from scipy.special import gamma
from Model_objs import *
#import warnings
arcsec2rad = (np.pi/(180.*3600.))
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.

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
            reff, index = source.reff['value'], source.index['value']
            dX = (xsource-xs)*np.cos(PA) + (ysource-ys)*np.sin(PA)
            dY = (-(xsource-xs)*np.sin(PA) + (ysource-ys)*np.cos(PA))/ar
            R = np.sqrt(dX**2. + dY**2.)
            
            # Calculate b_n, to make reff enclose half the light; this approx from Ciotti&Bertin99
            # This approximation good to 1 in 10^4 for n > 0.36; for smaller n it gets worse rapidly!!
            #if index < 0.35: warnings.warn("Sersic index n < 0.35 -- approximation to b_n is very bad in this regime!")
            bn = 2*index - 1./3. + 4./(405*index) + 46./(25515*index**2) + 131./(1148175*index**3) - 2194697./(30690717750*index**4)
            
            # Backing out from the integral to R=inf of a general sersic profile
            Ieff = source.flux['value'] * bn**(2*index) / (2*np.pi*reff**2 * ar * np.exp(bn) * index * gamma(2*index))
            
            return Ieff * np.exp(-bn*((R/reff)**(1./index)-1.))
      
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
