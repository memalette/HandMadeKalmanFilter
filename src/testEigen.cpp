#include "HMKF.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// [[Rcpp::export]]
SEXP test01(){
  MatrixXd m(2,2);
  m = MatrixXd::Zero(2,2);
  return wrap(m);
}

// [[Rcpp::export]]
SEXP testHMKF01(){
  HMKF kf(1,1);
  MatrixXd v0(1,1); v0(0,0) = 1;
  VectorXd x0(1);   x0[0] = 10;
  kf.setModel(v0,v0,v0,v0);
  kf.initialize(x0, v0);
  kf.predict();
  VectorXd y(1);    y(0) = 10;
  kf.filter(y);
  Eigen::VectorXd vp = kf.getxp();
  VectorXd v = kf.getx();
  return List::create(_("v") = v[0], _("vp") = vp[0]);
}

// [[Rcpp::export]]
SEXP testConvertFromArrayToVectorXd(NumericVector x){
  if(x.size() > 0){
    size_t size_vec = x.size();
    Eigen::Map<Eigen::VectorXd > mapxe(&x[0], size_vec);
    Eigen::VectorXd xe = mapxe;
    // mapxe
    std::cout << "mapxe:";
    for(int i=0; i!= size_vec; ++i){
      std::cout << mapxe(i) << " ";
    }
    std::cout << std::endl;
    // xe
    std::cout << "xe:";
    for(int i=0; i!= size_vec; ++i){
      xe(i) += 10;
      std::cout << xe(i) << " ";
    }
    std::cout << std::endl;
    // mapxe
    std::cout << "mapxe:";
    for(int i=0; i!= size_vec; ++i){
      std::cout << mapxe(i) << " ";
    }
    std::cout << std::endl;
  }
  return wrap(0);
}
