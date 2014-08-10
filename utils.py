import numpy as np
import scipy.stats as stats
from scipy.special import erfinv
from astropy.stats import sigma_clip
import copy
from Data_objs import Visdata
arcsec2rad = np.pi/180/3600.

__all__ = ['read_visdata','read_image','cart2pol','pol2cart','concatvis','bin_visibilities',\
            'expsinc','box']

def read_visdata(filename):
      """
      Function to read in visibility data from file and create a visdata object
      to hold it afterwards. So far only .bin files from get_visibilities.py are
      supported; idea is eventually to be able to not mess with that and get straight
      from a CASA ms, but don't currently know how to do that without bundling the 
      casacore utilities directly...

      Params:
      filename
            Name of file to read from. Should contain all the visibility data needed,
            including u (Lambda), v (Lambda), real, imag, sigma, antenna1, and antenna 2.

      Returns:
      visdata
            A visdata object containing the data from filename.
      """
      
      if not filename.split('.')[-1].lower() in ['bin']:
            raise ValueError('Only .bin files are supported for now...')

      data = np.fromfile(filename)
      PBfwhm = data[-1]
      data = data[:-1]
      data = data.reshape(7,data.size/7) # bin files lose array shape, so reshape to match

      return Visdata(*data,PBfwhm=PBfwhm,filename=filename)

def read_image(image,psf,noisemap=None,mask=None):
      """
      Function to read in an image and create an ImageData object to hold it.
      
      Params:
      image 
            numpy array, or name of .fits file to read from. Contains the actual image.
      
      psf
            numpy array, or name of .fits file to read from. Contains PSF image.
      
      noisemap
            numpy array, or name of .fits file to read from. An optional error map
            on the image data.  If None, the ImageData noisemap is initialized to be
            sqrt(image).
      
      mask
             numpy array, or name of .fits file to read from. An optional pixel
             mask. Should be same size as image, where non-zero values will be
             considered bad/masked. If None, initialized as zeros (no mask).
            
      Returns:
      ImageData
            An ImageData object.
      """
      
      return None

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

def concatvis(visdatas):
      """
      Concatenate multiple visibility sets into one larger set.
      Does no consistency checking of any kind, so beware.
      
      :param visdatas:
            List of visdata objects

      This method returns:

      * ``concatvis'' - The concatenated visibility set.
      """

      newu, newv, newr, newi = np.array([]),np.array([]),np.array([]),np.array([])
      news, newa1,newa2 = np.array([]),np.array([]),np.array([])

      for vis in visdatas:
            newu = np.concatenate((newu,vis.u))
            newv = np.concatenate((newv,vis.v))
            newr = np.concatenate((newr,vis.real))
            newi = np.concatenate((newi,vis.imag))
            news = np.concatenate((news,vis.sigma))
            newa1= np.concatenate((newa1,vis.ant1))
            newa2= np.concatenate((newa2,vis.ant2))

      return Visdata(newu,newv,newr,newi,news,newa1,newa2)      

def bin_visibilities(visdata,maxnewsize=None):
      """
      WARNING: DOESN'T WORK CURRENTLY(?)
      Bins up (ie, averages down) visibilities to reduce the total
      number of them.  Note that since we fit directly to the visibilities,
      this is slightly different (and easier) than gridding in preparation for
      imaging, as we won't need to FFT and so don't need a convolution function.

      :param visdata
            A Visdata object.
      :param maxnewsize = None
            If desired, the maximum number of visibilities post-binning can
            be specified. As long as this number meets other criteria (ie,
            we don't have bin sizes smaller than an integration time or
            bandwidth in wavelengths), the total number in the returned
            Visdata will have fewer than maxnewsize visibilities.

      This method returns:
      * ``BinnedVisibilities'' - A Visdata object containing binned visibilities.
      """

      if maxnewsize is None: maxnewsize = visdata.u.size/2

      # Bins should be larger than an integration; strictly only valid for an EW array,
      # and assumes a 20s integration time. Thus, this is a conservative estimate.
      minbinsize = 20. * visdata.uvdist.max() / (24*3600.)

      # Bins should be smaller than the effective field size
      maxbinsize = (visdata.PBfwhm * arcsec2rad)**-1

      print minbinsize,maxbinsize

      # We're going to find a binning solution iteratively; this gets us set up
      Nbins, binsizeunmet, Nvis, it, maxiter = [3000,3000], True, visdata.u.size, 0, 250
      
      while (binsizeunmet or Nvis >= maxnewsize):
            print Nbins
            # Figure out how to bin up the data
            counts,uedges,vedges,bins = stats.binned_statistic_2d(
                  visdata.u,visdata.v,values=visdata.real,statistic='count',
                  bins=Nbins)
            
            du, dv = uedges[1]-uedges[0], vedges[1]-vedges[0]

            # Check that our bins in u and v meet our conditions
            if (du > minbinsize and du < maxbinsize and
                dv > minbinsize and dv < maxbinsize): binsizeunmet = False
            # Otherwise we have to adjust the number of bins to adjust their size...
            #elif (du <= minbinsize or dv <= minbinsize): Nbins = int(Nbins/1.2)
            #elif (du >= maxbinsize or dv >= maxbinsize): Nbins = int(Nbins*1.2)
            elif du <= minbinsize: Nbins[0] = int(Nbins[0]/1.1); binsizeunmet=True
            elif dv <= minbinsize: Nbins[1] = int(Nbins[1]/1.1); binsizeunmet=True
            elif du >= maxbinsize: Nbins[0] = int(Nbins[0]*1.1); binsizeunmet=True
            elif dv >= maxbinsize: Nbins[1] = int(Nbins[1]*1.1); binsizeunmet=True

            # If we still have more than the desired number of visibilities, make
            # fewer bins (we'll loop after this).
            if np.unique(bins).size > maxnewsize: Nbins[0],Nbins[1] = int(Nbins[0]/1.1),int(Nbins[1]/1.1)
            Nvis = np.unique(bins).size
            it += 1
            if it > maxiter: raise ValueError("It's impossible to split your data into that few bins!  "
                                    "Try setting maxnewsize to a larger value!")
            print Nvis,du,dv


      # Get us some placeholder arrays for the binned data
      u,v,real,imag,sigma,ant1,ant2 = np.zeros((7,Nvis))

      for i,filledbin in enumerate(np.unique(bins)):
            # This tells us which visibilities belong to the current bin
            points = np.where(bins==filledbin)[0]
            # This unravels the indices to uedges,vedges from the binned_statistic binnumber
            uloc = int(np.floor(filledbin/(vedges.size+1)) - 1)
            vloc = int(filledbin - (vedges.size+1)*(uloc+1) - 1)
            # Get our new data, place at center of uv bins
            u[i],v[i] = uedges[uloc]+0.5*du, vedges[vloc]+0.5*dv
            real[i],sumwt = np.average(visdata.real[points],weights=visdata.sigma[points]**-2.,returned=True)
            imag[i] = np.average(visdata.imag[points],weights=visdata.sigma[points]**-2.)
            sigma[i] = sumwt**-0.5
            # We can keep the antenna numbers if we've only selected points from the same baseline,
            # otherwise get rid of them (CHECK IF MODELCAL FAILS WITH None ANTENNAS)
            ant1[i] = visdata.ant1[points][0] if (visdata.ant1[points]==visdata.ant1[points][0]).all() else None
            ant2[i] = visdata.ant2[points][0] if (visdata.ant2[points]==visdata.ant2[points][0]).all() else None

      return Visdata(u,v,real,imag,sigma,ant1,ant2,visdata.PBfwhm,'BIN{0}'.format(Nvis)+visdata.filename)


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


