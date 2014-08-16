import numpy as np
import scipy.sparse
from Data_objs import Visdata
import copy

def model_cal(realdata,modeldata,dPhi_dphi=None,FdPC=None):
      """
      Routine following Hezaveh+13 to implement perturbative phase corrections
      to model visibility data. This routine is designed to start out from an
      intermediate step in the self-cal process, since we can avoid doing a
      lot of expensive matrix inversions that way, but if either of the
      necessary arrays aren't provided, we calculate them.

      Inputs:
      realdata,modeldata:
            visdata objects containing the actual data and model-generated data

      dPhi_dphi: None, or pre-calculated
            See H+13, App A. An N_ant-1 x M_vis matrix whose ik'th element is 1
            if the first antenna of the visibility is k, -1 if the second, or 0.

      FdPC: None, or pre-calculated
            See H+13, App A, eq A2. This is an N_ant-1 x M_vis matrix, equal to
            -inv(F)*dPhi_dphi*inv(C) in the nomenclature of H+13. This has the 
            matrix inversions that we want to avoid calculating at every MCMC
            iteration (inverting C takes ~3s for M~5k, even with a sparse matrix).

      Outputs:
      modelcaldata:
            visdata object containing updated visibilities

      dphi:
            Array of length N_ant-1 containing the implemented phase offsets
      """

      # If we don't have the pre-calculated arrays, do so now. It's expensive
      # to do these matrix inversions at every MCMC step.
      if np.any((FdPC is None),(dPhi_dphi is None)):
            uniqant = np.unique(np.asarray([realdata.ant1,realdata.ant2]).flatten())
            dPhi_dphi = np.zeros((uniqant.size-1,realdata.u.size))
            for j in range(1,uniqant.size):
                  dPhi_dphi[j-1,:] = (realdata.ant1==uniqant[j])-1*(realdata.ant2==uniqant[j])
            C = scipy.sparse.diags((realdata.sigma/realdata.amp)**-2.,0)
            F = np.dot(dPhi_dphi,C*dPhi_dphi.T)
            Finv = np.linalg.inv(F)
            FdPC = np.dot(-Finv,dPhi_dphi*C)

      # Calculate current phase difference between data and model; wrap to +/- pi
      deltaphi = realdata.phase - modeldata.phase
      deltaphi = (deltaphi + np.pi) % (2 * np.pi) - np.pi

      dphi = np.dot(FdPC,deltaphi)

      modelcaldata = copy.deepcopy(realdata)
      
      modelcaldata.phase += np.dot(dPhi_dphi.T,dphi)

      return modelcaldata,dphi
