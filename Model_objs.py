# Collection of classes for various useful lenses/sources/shear
import numpy as np
__all__ = ['SIELens','SersicSource','GaussSource','PointSource',\
            'ExternalShear']

class SIELens:
      """
      Class to hold parameters for an SIE lens. Each is going to be a dictionary,
      holding the 'value' of the parameter, whether or not it should be 'fixed',
      and a lower and upper bound 'prior'. Only exception is redshift, assumed
      not to be a free parameter in any fitting procedure.

      Units of parameters are M: Msol; x,y: arcsec; e: 0-1; PA: degrees
      """      
      def __init__(self,z,x,y,M,e,PA):
            # Do some input handling.
            if not isinstance(x,dict):
                  x = {'value':x,'fixed':False,'prior':[-30.,30.]}
            if not isinstance(y,dict):
                  y = {'value':y,'fixed':False,'prior':[-30.,30.]}
            if not isinstance(M,dict):
                  M = {'value':M,'fixed':False,'prior':[1e7,1e15]}
            if not isinstance(e,dict):
                  e = {'value':e,'fixed':False,'prior':[0.,1.]}
            if not isinstance(PA,dict):
                  PA = {'value':PA,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [x,y,M,e,PA]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in x: x['fixed'] = False
            if not 'fixed' in y: y['fixed'] = False
            if not 'fixed' in M: M['fixed'] = False  
            if not 'fixed' in e: e['fixed'] = False
            if not 'fixed' in PA: PA['fixed'] = False
            
            if not 'prior' in x: x['prior'] = [-30.,30.]
            if not 'prior' in y: y['prior'] = [-30.,30.]
            if not 'prior' in M: M['prior'] = [1e7,1e15]
            if not 'prior' in e: e['prior'] = [0.,1.]
            if not 'prior' in PA: PA['prior'] = [0.,180.]

            self.z = z
            self.x = x
            self.y = y
            self.M = M
            self.e = e
            self.PA = PA

class SersicSource(object):
      """
      Create a symmetric Sersic profile object, with 
      I(x,y) = A * exp(-bn*((r/reff)^(1/n)-1)),
      where n is the Sersic index (0.5=gaussian,1=expdisk,4=elliptical),
      reff encloses half the light, and r=sqrt(x**2+y**2). To make a non-
      axisymmetric profile, we rescale x and y above to elliptical coords.
      bn is just a number to make reff enclose half the light.

      Specified here by the integrated flux (NOT peak amplitude). This helps
      avoid problems for large indices (even n~4), where tons of flux are
      spread to large distances.
      
      If lensed is True, it's assumed this source is in the background, and we
      lens it; otherwise it's assumed to be a foreground/unlensed object.
      
      Units of parameters are xoff,yoff: arcsec; flux: Jy; 
            width: reff in arcsec; axisratio: minor/major; PA: deg CCW from x-axis
      
      Note: if `lensed' is True, the positions are relative to the lens center,
            otherwise they're relative to the field center (or (0,0) coordinates)
      """

      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,reff=None,\
                  index=None,axisratio=None,PA=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
            if not isinstance(reff,dict):
                  reff = {'value':reff,'fixed':False,'prior':[0.,2.]} # arcsec
            if not isinstance(index,dict):
                  index = {'value':index,'fixed':False,'prior':[0.3,4.]}
            if not isinstance(axisratio,dict):
                  axisratio = {'value':axisratio,'fixed':False,'prior':[0.01,1.]}
            if not isinstance(PA,dict):
                  PA = {'value':PA,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [xoff,yoff,flux,reff,index,axisratio,PA]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            if not 'fixed' in reff: reff['fixed'] = False
            if not 'fixed' in index: index['fixed'] = False
            if not 'fixed' in axisratio: axisratio['fixed'] = False
            if not 'fixed' in PA: PA['fixed'] = False
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]
            if not 'prior' in reff: reff['prior'] = [0.,2.]
            if not 'prior' in index: index['prior'] = [1/3.,10]
            if not 'prior' in axisratio: axisratio['prior'] = [0.01,1.]
            if not 'prior' in PA: PA['prior'] = [0.,180.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux
            self.reff = reff
            self.index = index
            self.axisratio = axisratio
            self.PA = PA


class GaussSource(object):
      """
      Convenience wrapper to create a symmetric Gaussian source profile, a special
      case of a Sersic profile with n=0.5 and alpha=sqrt(2)*sigma, axis ratio = 1., PA=0.
      
      If lensed is True, it's assumed this source is in the background, and we
      lens it; otherwise it's assumed to be a foreground/unlensed object.

      Units of parameters are xoff,yoff: arcsec; flux: Jy; width: Gaussian rms in arcsec
      
      Note: if `lensed' is True, the positions are relative to the lens center,
            otherwise they're relative to the field center (or (0,0) coordinates)
      """

      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,width=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
            if not isinstance(width,dict):
                  width = {'value':width,'fixed':False,'prior':[0.,2.]} # arcsec

            if not all(['value' in d for d in [xoff,yoff,flux,width]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            if not 'fixed' in width: width['fixed'] = False
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]
            if not 'prior' in width: width['prior'] = [0.,2.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux
            self.width = width
            
class PointSource(object):
      """
      Creates an object which is unresolved by the data.
      
      The three parameters are the position (xoff,yoff), and integrated flux
      (flux).
      
      If lensed is True, it's assumed this source is in the background, and we
      lens it; otherwise it's assumed to be a foreground/unlensed object.

      Units of parameters are xoff,yoff: arcsec; flux: Jy
      
      Note: if `lensed' is True, the positions are relative to the lens center,
            otherwise they're relative to the field center (or (0,0) coordinates)
      """
      
      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source

            if not all(['value' in d for d in [xoff,yoff,flux]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux

        
class ExternalShear:
      """
      Class to hold the two parameters relating to external shear.

      Units are shear: 0-1; shearangle: degrees rel to lens PA.
      """
      def __init__(self,shear,shearangle):
            # Do some input handling.
            if not isinstance(shear,dict):
                  shear = {'value':shear,'fixed':False,'prior':[0.,1.]}
            if not isinstance(shearangle,dict):
                  shearangle = {'value':shearangle,'fixed':False,'prior':[0.,180.]}

            if not all(['value' in d for d in [shear,shearangle]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in shear: shear['fixed'] = False
            if not 'fixed' in shearangle: shearangle['fixed'] = False

            if not 'prior' in shear: shear['prior'] = [0.,1.]
            if not 'prior' in shearangle: shearangle['prior'] = [0.,180.]

            self.shear = shear
            self.shearangle = shearangle

