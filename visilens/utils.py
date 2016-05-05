import numpy as np
import scipy.stats as stats
arcsec2rad = (np.pi/(180.*3600.))

__all__ = ['cart2pol','pol2cart','expsinc','box','sersic_area']

def cart2pol(x,y):
    """
    Convert a set of x,y Cartesian coordinates. Why isn't this in numpy or scipy already?

    Inputs:
        x,y: Arrays or lists of coordinate pairs to transform

    Returns:
        r,theta: Arrays of polar coordinates from the given x and y, theta in radians
    """

    x, y = np.asarray(x), np.asarray(y)

    r = np.sqrt(x**2. + y**2.)
    theta = np.arctan2(y,x)

    return r, theta

def pol2cart(r,theta):
    """
    Convert a set of r,theta polar coordinates to Cartesian.

    Inputs:
        r,theta: Arrays or lists of coordinate pairs to transform; assumes theta in radians

    Returns:
        x,y: Arrays of Cartesian coordinates from the given r and theta
    """

    r, theta = np.asarray(r), np.asarray(theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x,y


def box(u,v,deltau,deltav):
      """
      Define a box/tophat/pillbox shaped convolution kernel. Bad for
      reducing aliasing.
      """
      
      try: # Assumes u is already a 2D grid of values
            conv = np.zeros(u.shape)
            conv[u.shape[0]/2.,u.shape[1]/2.] = 1.
      except: # Assumes u and v are still 1D coords
            conv = np.zeros((u.size,v.size))
            conv[u.size/2.,v.size/2.] = 1.
      
      return conv

def expsinc(u,v,deltau,deltav):
      """
      Define a sinc*gaussian exponential convolution kernel, following
      TMS Ch. 10.
      """

      alpha1 = 1.55 # See TMS, who cited... I forget, somebody.
      alpha2 = 2.52

      return np.sinc(u/(alpha1*deltau))*np.exp(-(u/(alpha2*deltau))**2.) * \
             np.sinc(v/(alpha1*deltav))*np.exp(-(v/(alpha2*deltav))**2.)

def sersic_area(n,majaxis,axisratio):
      """
      Calculate the total area (to R=infinity) of a Sersic profile.
      Uses the Ciotti+Bertin 1999 approx to b_n.

      :param n
            Sersic profile index. C&B1999 approx is good down to n~0.2

      :param majaxis
            Major axis of the profile, arbitrary units.

      :param axisratio
            Ratio of minor/major axis. If >1, we use 1/axisratio instead.

      This method returns:

      * ``sersicarea'' - The profile area, in square-majaxis units.
      """
      from scipy.special import gamma
      # C&B1999 eq 18
      bn = 2*n - 1./3. + 4./(405.*n) + 46./(25515*n**2) + 131./(1148175*n**3) - 2194697./(30690717750*n**4)
      # Graham&Driver 2005, eq 2. Note: exp(bn) term is due to using r_half [==r_eff] instead of r/r0, cf C&B1999 eq 4
      return 2*np.pi*n*majaxis**2 * axisratio * np.exp(bn) * gamma(2*n) / bn**(2*n)