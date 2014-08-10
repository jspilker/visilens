import numpy as np

class Visdata(object):
      """
      Class to hold all necessary info relating to one set of visibilities.
      Auto-updates amp&phase or real&imag if those values are changed, but
      MUST SET WITH, eg, visobj.amp = (a numpy array of the new values);
      CANNOT USE, eg, visobj.amp[0] = newval, AS THIS DOES NOT CALL THE
      SETTER FUNCTIONS.
      """
      
      def __init__(self,u,v,real,imag,sigma,ant1=None,ant2=None,PBfwhm=None,filename=None):
            self.u = u
            self.v = v
            self.real = real
            self.imag = imag
            self.sigma = sigma
            self.ant1 = ant1
            self.ant2 = ant2
            self.PBfwhm = PBfwhm
            self.filename = filename
      
      @property
      def uvdist(self):
            return np.sqrt(self.u**2. + self.v**2.)

      @property
      def real(self):
            return self._real
      @real.setter
      def real(self,val):
            self._real = val
            # Setting amp & phase during __init__ will fail since imag is still unknown
            # Doing so during conjugate() will also fail, but gives a ValueError
            try:
                  self._amp = np.sqrt(self._real**2. + self.imag**2.)
                  self._phase = np.arctan2(self.imag,self._real)
            except (AttributeError,ValueError):
                  self._amp = None
                  self._phase = None
            
      @property
      def imag(self):
            return self._imag
      @imag.setter
      def imag(self,val):
            self._imag = val
            try:
                  self._amp = np.sqrt(self.real**2. + self._imag**2.)
                  self._phase = np.arctan2(self._imag,self.real)
            except (AttributeError,ValueError):
                  self._amp = None
                  self._phase = None

      @property 
      def amp(self):
            return  self._amp
      @amp.setter
      def amp(self,val):
            self._amp = val
            self._real = val * np.cos(self.phase)
            self._imag = val * np.sin(self.phase)

      @property
      def phase(self):
            return self._phase
      @phase.setter
      def phase(self,val):
            self._phase = val
            self._real = self.amp * np.cos(val)
            self._imag = self.amp * np.sin(val)

      def __add__(self,other):
            return Visdata(self.u,self.v,self.real+other.real,self.imag+other.imag,\
                  (self.sigma**-2. + other.sigma**-2.)**-0.5)

      def __sub__(self,other):
            return Visdata(self.u,self.v,self.real-other.real,self.imag-other.imag,\
                  (self.sigma**-2. + other.sigma**-2.)**-0.5)

      def conjugate(self):
            u = np.concatenate((self.u,-self.u))
            v = np.concatenate((self.v,-self.v))
            real = np.concatenate((self.real,self.real))
            imag = np.concatenate((self.imag,-self.imag))
            sigma = np.concatenate((self.sigma,self.sigma))
            ant1 = np.concatenate((self.ant1,self.ant2))
            ant2 = np.concatenate((self.ant2,self.ant1))
            self.u = u
            self.v = v
            self.real = real
            self.imag = imag
            self.sigma = sigma
            self.ant1 = ant1
            self.ant2 = ant2

class ImageData(object):
      """
      Class to hold info relating to images. Simpler than visibility data.
      Each parameter should be a 2D array (even the x&y coordinates). Also allows
      for a mask to be passed, where non-zero points in the mask are considered bad.
      
      If you don't pass a noise map, one will be created as sqrt(data).
      """

      def __init__(self,x,y,data,header=None,sigma=None,mask=None,filename=None):
            if noisemap is None:
                  noisemap = np.sqrt(np.abs(np.asarray(data)))

            if mask is None:
                  mask = np.zeros(x.shape)

            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.header = header
            self.data = np.asarray(data)
            self.noisemap = np.asarray(noisemap)
            self.mask = np.asarray(mask)
            self.filename = filename

      def __add__(self,other):
            if isinstance(other,ImageData):
                  return ImageData(self.x,self.y,self.data+other.data,self.header,self.sigma,self.mask,self.filename)
            else: return ImageData(self.x,self.y,self.data+other,self.header,self.sigma,self.mask,self.filename)

      def __subtract__(self,other):
            if isinstance(other,ImageData):
                  return ImageData(self.x,self.y,self.data-self.other,self.header,self.sigma,self.mask,self.filename)
            else: return ImageData(self.x,self.y,self.data-other,self.header,self.sigma,self.mask,self.filename)

      def __mul__(self,other):
            if isinstance(other,ImageData):
                  return ImageData(self.x,self.y,self.data*self.other,self.header,self.sigma,self.mask,self.filename)
            else: return ImageData(self.x,self.y,self.data*other,self.header,self.sigma,self.mask,self.filename)

      def __div__(self,other):
            if isinstance(other,ImageData):
                  return ImageData(self.x,self.y,self.data/self.other,self.sigma,self.mask)
            else: return ImageData(self.x,self.y,self.data/np.asarray(other),self.sigma,self.mask)

      def __truediv__(self,other):
            if isinstance(other,ImageData):
                  return ImageData(self.x,self.y,self.data/self.other,self.sigma,self.mask)
            else: return ImageData(self.x,self.y,self.data/np.asarray(other),self.sigma,self.mask)

      
