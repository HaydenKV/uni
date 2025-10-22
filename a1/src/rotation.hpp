#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 8
    const Scalar cx = cos(x), sx = sin(x);
    R << Scalar(1), Scalar(0), Scalar(0),
         Scalar(0),      cx   ,     -sx   ,
         Scalar(0),      sx   ,      cx   ;
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 8
    const Scalar cy = cos(x), sy = sin(x);
    R <<      cy   , Scalar(0),      sy  ,
         Scalar(0), Scalar(1), Scalar(0) ,
            -sy   , Scalar(0),      cy   ;
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 8
    const Scalar cz = cos(x), sz = sin(x);
    R <<      cz   ,    -sz   , Scalar(0),
              sz   ,     cz   , Scalar(0),
         Scalar(0), Scalar(0), Scalar(1);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> rpy2rot(const Eigen::MatrixBase<Derived> & Theta)
{
    using Scalar = typename Derived::Scalar;
    using std::sin; using std::cos;
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 8
    const Scalar phi   = Theta(0); // roll
    const Scalar theta = Theta(1); // pitch
    const Scalar psi   = Theta(2); // yaw

    const Scalar cphi = cos(phi),   sphi = sin(phi);
    const Scalar cth  = cos(theta), sth  = sin(theta);
    const Scalar cpsi = cos(psi),   spsi = sin(psi);

    R <<  cpsi*cth,                 cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi,
          spsi*cth,                 spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi,
              -sth,                                  cth*sphi,                   cth*cphi;

    return R;
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> rot2rpy(const Eigen::MatrixBase<Derived> & R)
{
    using Scalar = typename Derived::Scalar;
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta; // [phi; theta; psi] = [roll; pitch; yaw]
    // TODO: Lab 8

    // ZYX convention:
    // theta = atan2(-R(2,0), sqrt(R(0,0)^2 + R(1,0)^2))
    const Scalar r20 = R(2,0);
    const Scalar r00 = R(0,0);
    const Scalar r10 = R(1,0);
    const Scalar ctheta = hypot(r00, r10);  // = |cos(theta)|

    const Scalar theta = atan2(-r20, ctheta);

    // phi = atan2(R(2,1), R(2,2))
    const Scalar phi = atan2(R(2,1), R(2,2));

    // psi = atan2(R(1,0), R(0,0))
    const Scalar psi = atan2(r10, r00);

    Theta << phi, theta, psi;
    return Theta;
}

#endif
