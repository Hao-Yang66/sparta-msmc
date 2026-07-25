/* ----------------------------------------------------------------------
   SPARTA-MSMC V1.0
   An efficient particle solver for continuum-to-rarefied gas flows
   based on the Multiscale Simulation Monte Carlo method.

   Developer: Hao Yang
   Organization: Beihang University
   Email: yang_hao@buaa.edu.cn

   Minimal math helpers for MSMC polyatomic coefficient evaluation.
------------------------------------------------------------------------- */

#ifndef SPARTA_MSMC_MATH_H
#define SPARTA_MSMC_MATH_H

#include <cmath>
#include <algorithm>

namespace msmc_math {

struct Vec3 {
  double v[3];

  Vec3() { v[0] = v[1] = v[2] = 0.0; }
  Vec3(double v0, double v1, double v2) { v[0] = v0; v[1] = v1; v[2] = v2; }

  Vec3 operator+(const Vec3 &rhs) const {
    return Vec3(v[0] + rhs.v[0], v[1] + rhs.v[1], v[2] + rhs.v[2]);
  }

  Vec3 operator-(const Vec3 &rhs) const {
    return Vec3(v[0] - rhs.v[0], v[1] - rhs.v[1], v[2] - rhs.v[2]);
  }

  Vec3 operator*(double s) const {
    return Vec3(v[0] * s, v[1] * s, v[2] * s);
  }
};

inline Vec3 operator*(double s, const Vec3 &rhs) { return rhs * s; }

inline double norm_inf(const Vec3 &rhs)
{
  return std::max(std::max(std::fabs(rhs.v[0]), std::fabs(rhs.v[1])),
                  std::fabs(rhs.v[2]));
}

struct Mat3x3 {
  double m[3][3];

  Mat3x3() {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        m[i][j] = 0.0;
  }

  Mat3x3(double m00, double m01, double m02,
         double m10, double m11, double m12,
         double m20, double m21, double m22)
  {
    m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
    m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
    m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
  }

  static Mat3x3 Identity() {
    Mat3x3 I;
    I.m[0][0] = I.m[1][1] = I.m[2][2] = 1.0;
    return I;
  }

  Mat3x3 operator+(const Mat3x3 &rhs) const {
    Mat3x3 out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out.m[i][j] = m[i][j] + rhs.m[i][j];
    return out;
  }

  Mat3x3 operator-(const Mat3x3 &rhs) const {
    Mat3x3 out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out.m[i][j] = m[i][j] - rhs.m[i][j];
    return out;
  }

  Mat3x3 operator*(double s) const {
    Mat3x3 out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out.m[i][j] = m[i][j] * s;
    return out;
  }

  Mat3x3 operator*(const Mat3x3 &rhs) const {
    Mat3x3 out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out.m[i][j] = m[i][0] * rhs.m[0][j] +
                      m[i][1] * rhs.m[1][j] +
                      m[i][2] * rhs.m[2][j];
    return out;
  }

  Vec3 operator*(const Vec3 &rhs) const {
    return Vec3(m[0][0] * rhs.v[0] + m[0][1] * rhs.v[1] + m[0][2] * rhs.v[2],
                m[1][0] * rhs.v[0] + m[1][1] * rhs.v[1] + m[1][2] * rhs.v[2],
                m[2][0] * rhs.v[0] + m[2][1] * rhs.v[1] + m[2][2] * rhs.v[2]);
  }

  Mat3x3 transpose() const {
    Mat3x3 out;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out.m[i][j] = m[j][i];
    return out;
  }

  double determinant() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  Mat3x3 inverse() const {
    const double det = determinant();
    Mat3x3 inv;
    const double invdet = 1.0 / det;
    inv.m[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    inv.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    inv.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    inv.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    inv.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    inv.m[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    inv.m[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    inv.m[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    inv.m[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;
    return inv;
  }
};

inline Mat3x3 operator*(double s, const Mat3x3 &rhs) { return rhs * s; }

inline double norm_inf(const Mat3x3 &rhs)
{
  double out = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      out = std::max(out, std::fabs(rhs.m[i][j]));
  return out;
}

inline double solve_h(const double &C, const double &x,
                      const double &B, const double &D, const double &y,
                      double epsB = 1.0e-14, double epsR = 1.0e-14)
{
  const double rhs = D * y - C * x;
  if (std::fabs(B) > epsB) return rhs / B;
  if (std::fabs(rhs) <= epsR) return 0.0;
  return 0.0;
}

inline Vec3 solve_h(const Mat3x3 &C, const Vec3 &x,
                    const Mat3x3 &B, const Mat3x3 &D, const Vec3 &y,
                    double epsB = 1.0e-14, double epsR = 1.0e-14,
                    double reg0 = 1.0e-12)
{
  const Vec3 rhs = D * y - C * x;
  if (norm_inf(B) <= epsB && norm_inf(rhs) <= epsR) return Vec3();

  const Mat3x3 Bt = B.transpose();
  const Mat3x3 BtB = Bt * B;
  const Vec3 Btr = Bt * rhs;

  double scale = norm_inf(BtB);
  if (scale < 1.0) scale = 1.0;
  double lambda = reg0 * scale;

  for (int k = 0; k < 8; k++) {
    const Mat3x3 A = BtB + lambda * Mat3x3::Identity();
    if (std::fabs(A.determinant()) > epsB) return A.inverse() * Btr;
    lambda *= 10.0;
  }

  return Vec3();
}

}

#endif
