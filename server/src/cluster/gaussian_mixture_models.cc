#include "gaussian_mixture_models.h"

void CmGMM::reWeights(vector<double> &mulWs) {
  double sumW = 0;
  vector<double> newW(_K);
  for (int i = 0; i < _K; i++) {
    newW[i] = _Guassians[i].w * mulWs[i];
    sumW += newW[i];
  }
  for (int i = 0; i < _K; i++)
    _Guassians[i].w = newW[i] / sumW;

}
