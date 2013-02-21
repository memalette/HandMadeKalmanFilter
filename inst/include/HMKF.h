#ifndef GUARD_HMKF_H
#define GUARD_HMKF_H

#include <Rcpp.h>
#include <RcppEigen.h>

class HMKF{
public:
  HMKF(size_t dimState, size_t dimObservation)
  : x_(dimState), xp_(dimState), V_(dimState, dimState), Vp_(dimState, dimState)
  , F_(dimState, dimState), Ftrans_(dimState, dimState)
  , H_(dimObservation, dimState), Htrans_(dimState, dimObservation)
  , Q_(dimState, dimState)
  , R_(dimObservation, dimObservation)
  , S_(dimObservation, dimObservation)
  , K_(dimState, dimObservation)
  {};
  void setModel(Eigen::MatrixXd F, Eigen::MatrixXd H, Eigen::MatrixXd Q, Eigen::MatrixXd R){
    F_ = F; // system model linear coeffcient
//    Ftrans_ = F.transpose(); // worse performance 
    H_ = H; // observation model linear coefficient
//    Htrans_ = H.transpose(); // worse performance 
    Q_ = Q; // variance matrix for system model
    R_ = R; // variance matrix for observation model
  }
  void initialize(Eigen::VectorXd x0, Eigen::MatrixXd V0){
    x_ = x0;
    V_ = V0;
  }
  void predict(){
    xp_ = F_ * x_;
    Vp_ = F_ * V_ * F_.transpose() + Q_;
//    Vp_ = F_ * V_ * Ftrans_ + Q_;
  }
  void filter(Eigen::VectorXd y){
    S_ = H_ * Vp_ * H_.transpose() + R_;
//    S_ = H_ * Vp_ * Htrans_ + R_;
    K_ = Vp_ * H_.transpose() * S_.inverse(); 
//    K_ = Vp_ * Htrans_ * S_.inverse(); 
    x_ = xp_ + K_ * (y - H_ * xp_);
    V_ = Vp_ - K_ * H_ * Vp_;
  }
  Eigen::VectorXd getx(){ return x_;}
  Eigen::VectorXd getxp(){ return xp_;}
  Eigen::MatrixXd getV(){ return V_;} 
  Eigen::MatrixXd getVp(){ return Vp_;} 
private:
  Eigen::VectorXd x_, xp_, y_;
  Eigen::MatrixXd V_, Vp_;
  // constant model
  Eigen::MatrixXd F_,Ftrans_,H_,Htrans_,Q_,R_;
  // Kalman Gain
  Eigen::MatrixXd K_;
  // prediction error
  Eigen::MatrixXd S_;
};

#endif

 
