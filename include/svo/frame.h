// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_FRAME_H_
#define SVO_FRAME_H_

#include <sophus/se3.h>
#include <svo/math_lib.h>
#include <svo/camera_model.h>
#include <boost/noncopyable.hpp>
#include <svo/global.h>

#include "../src/IMU/IMUPreintegrator.h"
#include "../src/IMU/NavState.h"
#include "svo/Converter.h"
#include "../src/IMU/g2otypes.h"

namespace g2o {
class VertexSE3Expmap;
}
typedef g2o::VertexSE3Expmap g2oFrameSE3;
typedef g2o::VertexNavStatePVR  g2oNavStatePVR;
typedef g2o::VertexNavStateBias g2oNavStateBias;

namespace svo {

class Point;
struct Feature;

typedef list<Feature*> Features;
typedef vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
  static int                    frame_counter_;         //!< Counts the number of created frames. Used to set the unique id.
  int                           id_;                    //!< Unique id of the frame.
  int                           kf_id_;                 //!< gzh add
  int                           window_kf_id_;          //!< gzh add
  double                        timestamp_;             //!< Timestamp of when the image was recorded.
  //svo::AbstractCamera*           cam_;                   //!< Camera model.
  svo::PinholeCamera*           cam_;                   //!< Camera model.
  Sophus::SE3                   T_f_w_;                 //!< Transform (f)rame from (w)orld.从世界坐标到相机坐标的变换矩阵  Tcw
  Matrix<double, 6, 6>          Cov_;                   //!< Covariance.
  cv::Mat                       debug_img_;             // used to draw feature in img_pyr_[0]
  ImgPyr                        img_pyr_;               //!< Image Pyramid.
  Features                      fts_;                   //!< List of features in the image.
  vector<Feature*>              key_pts_;               //!< Five features and associated 3D points which are used to detect if two frames have overlapping field of view.
  bool                          is_keyframe_;           //!< Was this frames selected as keyframe?
  bool                          have_initializeSeeds;

  g2oFrameSE3*                  v_kf_;                  //!< Temporary pointer to the g2o node object of the keyframe.
  int                           last_published_ts_;     //!< Timestamp of last publishing.

  Frame*                        PrevKeyFrame;
  Frame*                        NextKeyFrame;

  // SE3 Pose and camera center
  cv::Mat Tcw; // ==T_f_w_
  cv::Mat Twc;
  cv::Mat Ow;  //camera in world frame

  void SetPrevKeyFrame(Frame* pKF);
  void SetNextKeyFrame(Frame* pKF);
  Frame* GetPrevKeyFrame();
  Frame* GetNextKeyFrame();

  ///-------------gzh: add imu part-------------
  NavState						imuState;   // b in w_frame
  IMUPreintegrator				imuPreint;
  IMUPreintegrator				kfimuPreint;

  vector<IMUData> kfIMUData; //imudata from last KF to cur KF
  vector<IMUData> Get_kfIMUData() { return kfIMUData; }
  void ComputePreInt();  //compute IMU preint between kfs

  const NavState& GetNavState();
  void SetNavStatePos(const Vector3d &pos);     //p
  void SetNavStateVel(const Vector3d &vel);     //v
  void SetNavStateRot(const Matrix3d &rot);     //R
  void SetNavStateRot(const Sophus::SO3 &rot);
  void SetNavStateBiasGyr(const Vector3d &bg);  //bg
  void SetNavStateBiasAcc(const Vector3d &ba);  //ba
  void SetNavStateDeltaBg(const Vector3d &dbg); //delta bg
  void SetNavStateDeltaBa(const Vector3d &dba);  //delta_ba
  void SetInitialNavStateAndBias(const NavState& ns);

  void UpdateNavState(const NavState& lastImuState, const IMUPreintegrator& imupreint, const Eigen::Vector3d& gw);

  void GetpreVelFromeV( FramePtr curframe, const Vector3d gravity, const cv::Mat &Tbc);
  void UpdateNavstateFromV( FramePtr prevframe, Vector3d gravity, const cv::Mat& Tbc);
  Sophus::SE3 UpdatePoseFromNS(const cv::Mat& Tbc);

  cv::Mat GetVPoseInverse();   // Twc:transform from c to w
  cv::Mat GetVPos();           // Pwc
  void SetVPose(const cv::Mat &tcw);
  void SetVPos(const Vector3d& pos ){ T_f_w_.translation() = pos; }
  ///-------------end---------------------------

  ///-------------gzh: add window BA part-------------
  g2oNavStatePVR*            v_PVR_;                //gzh add
  g2oNavStateBias*           v_Bias_;               //gzh add
  bool                       is_infixed_kf;         //gzh add
  bool                       is_inWindow_;          //gzh add
  int                        winmap_obs_ftr;        //the number of featrues,kf can observed (in window maps)
  /// put this kf to slidingwindow.
  void putKFtoWindow();            //gzh add 2019-9-3
  void putKFtofixedKF();           //gzh add 2019-9-6
    ///-------------end--

//  Frame(svo::AbstractCamera* cam, const cv::Mat& img, double timestamp);
   Frame(svo::PinholeCamera* cam, const cv::Mat& img, double timestamp);
  ~Frame();

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat& img);

  /// Select this frame as keyframe.
  void setKeyframe();

  /// Add a feature to the image
  void addFeature(Feature* ftr);

  /// The KeyPoints are those five features which are closest to the 4 image corners
  /// and to the center and which have a 3D point assigned. These points are used
  /// to quickly check whether two frames have overlapping field of view.
  void setKeyPoints();

  /// Check if we can select five better key-points.
  void checkKeyPoints(Feature* ftr);

  /// If a point is deleted, we must remove the corresponding key-point.
  void removeKeyPoint(Feature* ftr);

  /// Return number of point observations.
  inline size_t nObs() const { return fts_.size(); }

  /// Check if a point in (w)orld coordinate frame is visible in the image.
  bool isVisible(const Vector3d& xyz_w) const;

  /// Full resolution image stored in the frame.
  inline const cv::Mat& img() const { return img_pyr_[0]; }

  /// Was this frame selected as keyframe?
  inline bool isKeyframe() const { return is_keyframe_; }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
  inline Vector2d w2c(const Vector3d& xyz_w) const { return cam_->world2cam( T_f_w_ * xyz_w ); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const Vector2d& px) const { return cam_->cam2world(px[0], px[1]); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

  /// Transforms point coordinates in world-frame (w) to camera-frams (f).
  inline Vector3d w2f(const Vector3d& xyz_w) const { return T_f_w_ * xyz_w; }

  /// Transforms point from frame unit sphere (f) frame to world coordinate frame (w).
  inline Vector3d f2w(const Vector3d& f) const { return T_f_w_.inverse() * f; }

  /// Projects Point from unit sphere (f) in camera pixels (c).
  inline Vector2d f2c(const Vector3d& f) const { return cam_->world2cam( f ); }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }

  /// Return the pose of the frame in the (w)orld coordinate frame.//返回世界系下的 rotation
  inline Matrix3d rot() const { return T_f_w_.inverse().rotation_matrix(); }

  /// Frame jacobian for projection of 3D point in (f)rame coordinate to
  /// unit plane coordinates uv (focal length = 1).
  inline static void jacobian_xyz2uv(
      const Vector3d& xyz_in_f,
      Matrix<double,2,6>& J)
  {
    const double x = xyz_in_f[0];
    const double y = xyz_in_f[1];
    const double z_inv = 1./xyz_in_f[2];
    const double z_inv_2 = z_inv*z_inv;

    J(0,0) = -z_inv;              // -1/z
    J(0,1) = 0.0;                 // 0
    J(0,2) = x*z_inv_2;           // x/z^2
    J(0,3) = y*J(0,2);            // x*y/z^2
    J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
    J(0,5) = y*z_inv;             // y/z

    J(1,0) = 0.0;                 // 0
    J(1,1) = -z_inv;              // -1/z
    J(1,2) = y*z_inv_2;           // y/z^2
    J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
    J(1,4) = -J(0,3);             // -x*y/z^2
    J(1,5) = -x*z_inv;            // x/z
  }
};


/// Some helper functions for the frame object.
namespace frame_utils {

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

/// Get the average depth of the features in the image.
bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min);

} // namespace frame_utils
} // namespace svo

#endif // SVO_FRAME_H_
