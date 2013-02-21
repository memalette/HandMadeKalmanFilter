#include "HMKF.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <stdexcept>
#include <iostream>
using namespace Rcpp;
using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
SEXP HandMadeKalmanFilterConstantCoeffCpp(NumericVector a0
  , NumericMatrix P0, NumericMatrix T, NumericMatrix Z , NumericMatrix HH, NumericMatrix GG, NumericMatrix yt)
{
  // convert data to Eigen class
  Eigen::Map<Eigen::VectorXd > a0e(&a0[0], (size_t)(a0.size()));
  Eigen::Map<Eigen::VectorXd > P0e(&P0[0], P0.rows(), P0.cols());
  Eigen::Map<Eigen::MatrixXd > Te(&T[0], T.rows(), T.cols());
  Eigen::Map<Eigen::MatrixXd > Ze(&Z[0], Z.rows(), Z.cols());
  Eigen::Map<Eigen::MatrixXd > HHe(&HH[0], HH.rows(), HH.cols());
  Eigen::Map<Eigen::MatrixXd > GGe(&GG[0], GG.rows(), GG.cols());
  
  // HMKF initialization block
  cout << T.rows() << " " << Z.rows() << " " << HH.rows() << " " << GG.rows() << endl;
  HMKF kf = HMKF(T.rows(), Z.rows());
  kf.setModel(Te, Ze, HHe, GGe);
  kf.initialize(a0e, P0e);

  // filtering steps
  NumericVector x(yt.cols()), xp(yt.cols()), V(yt.cols());
  for(int i=0; i!=yt.cols(); ++i){
    kf.predict();
    NumericMatrix::Column col = yt(_,i);
//    std::cout << "y:" << col[0] << std::endl;
    kf.filter(Map<Eigen::VectorXd >(&col[0], col.size()));
//    xp[i]= (kf.getxp())(0);
    x[i] = (kf.getx())(0);
//    V[i] = (kf.getV())(0,0);
  }
  return List::create(_("x") = x, _("xp") = xp, _("V") = V);
}

