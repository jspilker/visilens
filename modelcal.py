import numpy as np
from Data_objs import Visdata
import copy

def model_cal(realdata,modeldata,dPhi_dphi,FdPC):
      """
      Routine following Hezaveh+13 to implement perturbative phase corrections
      to model visibility data. This routine is designed to start out from an
      intermediate step in the self-cal process, since we can avoid doing a
      lot of expensive matrix inversions that way.

      Inputs:
      realdata,modeldata:
            visdata objects containing the actual data and model-generated data

      dPhi_dphi:
            See H+13, App A. An N_ant-1 x M_vis matrix whose ik'th element is 1
            if the first antenna of the visibility is k, -1 if the second, or 0.

      FdPC:
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

      # Calculate current phase difference between data and model; wrap to +/- pi
      deltaphi = realdata.phase - modeldata.phase
      deltaphi = (deltaphi + np.pi) % (2 * np.pi) - np.pi

      dphi = np.dot(FdPC,deltaphi)

      modelcaldata = copy.deepcopy(realdata)
      
      modelcaldata.phase += np.dot(dPhi_dphi.T,dphi)

      return modelcaldata,dphi
