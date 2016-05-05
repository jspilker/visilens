import numpy as np
import scipy.sparse
import copy
from class_utils import *
from utils import *
from astropy.cosmology import WMAP9
import astropy.constants as co

c = co.c.value # speed of light, in m/s
G = co.G.value # gravitational constant in SI units
Msun = co.M_sun.value # solar mass, in kg
Mpc = 1e6*co.pc.value # 1 Mpc, in m
arcsec2rad = np.pi/(180.*3600.)
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.
rad2deg = 180./np.pi

__all__ = ['LensRayTrace','GenerateLensingGrid','thetaE','CausticsSIE']

def LensRayTrace(xim,yim,lens,Dd,Ds,Dds):
      """
      Wrapper to pass off lensing calculations to any number of functions
      defined below, accumulating lensing offsets from multiple lenses
      and shear as we go.
      """
      # Ensure lens is a list, for convenience
      lens = list(np.array([lens]).flatten())
      
      ximage = xim.copy()
      yimage = yim.copy()
      
      for i,ilens in enumerate(lens):
            if ilens.__class__.__name__ == 'SIELens': ilens.deflect(xim,yim,Dd,Ds,Dds)
            elif ilens.__class__.__name__ == 'ExternalShear': ilens.deflect(xim,yim,lens[0])
            ximage += ilens.deflected_x; yimage += ilens.deflected_y
      
      return ximage,yimage

def GenerateLensingGrid(data=None,xmax=None,emissionbox=[-5,5,-5,5],fieldres=None,emitres=None):
      """
      Routine to generate two grids for lensing. The first will be a lower-resolution
      grid with resolution determined by fieldres and size determined
      by xmax. The second is a much higher resolution grid which will be used for
      the lensing itself, with resolution determined by emitres and size
      determined from emissionbox - i.e., emissionbox should contain the coordinates
      which conservatively encompass the real emission, so we only have to lens that part
      of the field at high resolution.

      Since we're going to be FFT'ing with these coordinates, the resolution isn't
      directly set-able. For the low-res full-field map, it instead is set to the next-higher
      power of 2 from what would be expected from having ~4 resolution elements across
      the synthesized beam.

      Inputs:
      data:
            A Visdata object, used to determine the resolutions of 
            the two grids (based on the image size or maximum uvdistance in the dataset)
      xmax:
            Field size for the low-resolution grid in arcsec, which will extend from
            (-xmax,-xmax) to (+xmax,+xmax), e.g. (-30,-30) to (+30,+30)arcsec. Should be
            at least a bit bigger than the primary beam. Not needed for images.
      emissionbox:
            A 1x4 list of [xmin,xmax,ymin,ymax] defining a box (in arcsec) which contains
            the source emission.  Coordinates should be given in arcsec relative to the
            pointing/image center.
      fieldres,emitres:
            Resolutions of the coarse, full-field and fine (lensed) field, in arcsec.
            If not given, suitable values will be calculated from the visibilities.
            fieldres is unnecessary for images.

      Returns:
      If there are any Visdata objects in the datasets, returns:
      xmapfield,ymapfield:
            2xN matrices containing x and y coordinates for the full-field, lower-resolution
            grid, in arcsec.
      xmapemission,ymapemission:
            2xN matrices containing x and y coordinates for the smaller, very high resolution
            grid, in arcsec.
      indices:
            A [4x1] array containing the indices of xmapfield,ymapfield which overlap with
            the high resolution grid.
      """

      # Factors higher-resolution than (1/2*max(uvdist)) to make the field and emission grids
      Nover_field = 4.
      Nover_emission = 8.

      # Allow multiple visdata objects to be passed, pick the highest resolution point of all
      uvmax = 0.
      try:
            for vis in data:
                  uvmax = max(uvmax,vis.uvdist.max())
      except TypeError:
            uvmax = data.uvdist.max()

      # Calculate resolutions of the grids
      if fieldres is None: fieldres = (2*Nover_field*uvmax)**-1.
      else: fieldres *= arcsec2rad
      if emitres is None: emitres  = (2*Nover_emission*uvmax)**-1.
      else: emitres *= arcsec2rad

      # Calculate the field grid size as a power of 2.
      Nfield = 2**np.ceil(np.log2(2*np.abs(xmax)*arcsec2rad/fieldres))

      # Calculate the grid coordinates for the larger field.
      fieldcoords = np.linspace(-np.abs(xmax),np.abs(xmax),Nfield)
      xmapfield,ymapfield = np.meshgrid(fieldcoords,fieldcoords)

      # Calculate the indices where the high-resolution lensing grid meets the larger field grid
      indices = np.round(np.interp(np.asarray(emissionbox),fieldcoords,np.arange(Nfield)))

      # Calculate the grid coordinates for the high-res lensing grid; grids meet at indices. Some pixel-shifting reqd.
      Nemx = 1 + np.abs(indices[1]-indices[0])*np.ceil((fieldcoords[1]-fieldcoords[0])/(2*emitres*rad2arcsec))
      Nemy = 1 + np.abs(indices[3]-indices[2])*np.ceil((fieldcoords[1]-fieldcoords[0])/(2*emitres*rad2arcsec))
      xemcoords = np.linspace(fieldcoords[indices[0]],fieldcoords[indices[1]],Nemx)
      yemcoords = np.linspace(fieldcoords[indices[2]],fieldcoords[indices[3]],Nemy)
      xmapemission,ymapemission = np.meshgrid(xemcoords,yemcoords)
      xmapemission -= (xmapemission[0,1]-xmapemission[0,0])
      ymapemission -= abs((ymapemission[1,0]-ymapemission[0,0]))

      return xmapfield,ymapfield,xmapemission,ymapemission,indices
      
def thetaE(ML,zL,zS,cosmo=WMAP9):
      """
      Calculate the Einstein radius in arcsec of a lens of mass ML,
      assuming redshifts zL and zS. If cosmo is None, WMAP9
      is assumed. ML is in solar masses.
      """
      
      Dd = cosmo.angular_diameter_distance(zL).value # in Mpc
      Ds = cosmo.angular_diameter_distance(zS).value
      Dds= cosmo.angular_diameter_distance_z1z2(zL,zS).value
      
      thE = np.sqrt((4*G*ML*Msun*Dds) / (c**2 * Dd*Ds*Mpc)) * rad2arcsec
      
      return thE

def CausticsSIE(SIELens,Dd,Ds,Dds,Shear=None):
      """
      Routine to calculate and return the analytical solutions for the caustics
      of an SIE Lens, following Kormann+94.

      Inputs:
      SIELens:
            An SIELens object for which to calculate the caustics.

      Dd,Ds,Dds:
            Angular diameter distances to the lens, source, and lens-source, respectively.

      Shear:
            An ExternalShear object describing the shear of the lens.

      Returns:
      2xN list:
            Arrays containing the x and y coordinates for the caustics that exist (i.e.,
            will have [xr,yr] for the radial caustic only if lens ellipticity==0, otherwise
            will have [[xr,yr],[xt,yt]] for radial+diamond caustics)
      """

      # Following Kormann+ 1994 for the lensing. Easier to work with axis ratio than ellipticity
      f = 1. - SIELens.e['value']
      fprime = np.sqrt(1. - f**2.)

      # K+94 parameterize lens in terms of LOS velocity dispersion; calculate here in m/s
      sigma_lens = ((SIELens.M['value']*Ds*G*Msun*c**2.)/(4.*np.pi**2. * Dd*Dds*Mpc))**(1./4.)

      # Einstein radius, for normalizing the size of the caustics, b in notation of Keeton+00
      b = 4 * np.pi * (sigma_lens/c)**2. * (Dds/Ds) * rad2arcsec      

      # Caustics calculated over a full 0,2pi angle range
      phi = np.linspace(0,2*np.pi,2000)

      # K+94, eq 21c; needed for diamond caustic
      Delta = np.sqrt(np.cos(phi)**2. + f**2. * np.sin(phi)**2.)
      
      if ((Shear is None) or (np.isclose(Shear.shear['value'],0.))):
            # Need to account for when ellipticity=0, as caustic equations have cancelling infinities
            #  In that case, Delta==1 and there's only one (radial and circular) caustic
            if np.isclose(f,1.):
                  xr,yr = -b*np.cos(phi)+SIELens.x['value'], -b*np.sin(phi)+SIELens.y['value']
                  caustic = np.atleast_3d([xr,yr])
                  return caustic.reshape(caustic.shape[2],caustic.shape[0],caustic.shape[1])

            else:
                  # Calculate the radial caustic coordinates
                  xr = (b*np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f)
                  yr = (-b*np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime)
                  
                  # Now rotate & shift the caustic to match the PA & loc of the lens
                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  xr += SIELens.x['value']
                  yr += SIELens.y['value']

                  # Calculate the tangential caustic coordinates
                  xt = b*(((np.sqrt(f)/Delta) * np.cos(phi)) - ((np.sqrt(f)/fprime)*np.arcsinh(fprime/f * np.cos(phi))))
                  yt = -b*(((np.sqrt(f)/Delta) * np.sin(phi)) - ((np.sqrt(f)/fprime)*np.arcsin(fprime * np.sin(phi))))

                  # ... and rotate it to match the lens
                  r,th = cart2pol(xt,yt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  xt += SIELens.x['value']
                  yt += SIELens.y['value']

                  return np.atleast_3d([[xr,yr],[xt,yt]])

      else: # Blerg, complicated expressions... Keeton+00, but at least radial pseudo-caustic doesn't depend on shear       
            s, sa = Shear.shear['value'], (Shear.shearangle['value']-SIELens.PA['value'])*deg2rad

            if np.isclose(f,1.):
                  rcrit = b * (1.+s*np.cos(2*(phi-sa)))/(1.-s**2.)

                  xr = -b*np.cos(phi) + SIELens.y['value']
                  yr = b*np.sin(phi) - SIELens.x['value']

                  xt = (np.cos(phi) + s*np.cos(phi-2*sa))*rcrit + xr - SIELens.y['value']
                  yt = (np.sin(-phi) - s*np.sin(-phi+2*sa))*rcrit + yr + SIELens.x['value']

                  r,th = cart2pol(yt,xt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  xt += SIELens.x['value']
                  yt += SIELens.y['value']
                  
                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  
                  return np.atleast_3d([[xr,yr],[xt,yt]])

            else:
                  rcrit = np.sqrt(2.*f)*b*(1.+s*np.cos(2.*(phi-sa))) / ((1.-s**2.)*np.sqrt((1+f**2.) - (1-f**2.)*np.cos(2*phi)))

                  xi = np.sqrt((2*(1-f**2.)) / ((1+f**2.)-(1-f**2.)*np.cos(2*phi)))
                  xr = -(b*np.sqrt(f)/fprime)*np.arctanh(xi*np.sin(phi))
                  yr = (b*np.sqrt(f)/fprime)*np.arctan(xi*np.cos(phi))

                  xt = (np.sin(phi)-s*np.sin(phi-2*sa))*rcrit + xr
                  yt = (np.cos(np.pi-phi)+s*np.cos(np.pi-phi+2*sa))*rcrit + yr       

                  r,th = cart2pol(xt,yt)
                  xt,yt = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  xt += SIELens.x['value']
                  yt += SIELens.y['value']

                  r,th = cart2pol(xr,yr)
                  xr,yr = pol2cart(r,th+SIELens.PA['value']*deg2rad)
                  xr += SIELens.x['value']
                  yr += SIELens.y['value']

                  return np.atleast_3d([[xr,yr],[xt,yt]])