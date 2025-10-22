#ifndef  BALLISTIC_PLOT_H
#define  BALLISTIC_PLOT_H

#include <Eigen/Core>

void plot_simulation(
    const Eigen::VectorXd & t_hist, 
    const Eigen::MatrixXd & x_hist, 
    const Eigen::MatrixXd & mu_hist, 
    const Eigen::MatrixXd & sigma_hist);
    
#endif 
