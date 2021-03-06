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

#ifndef SVO_FRAME_HANDLER_H_
#define SVO_FRAME_HANDLER_H_

#include <set>
#include <svo/camera_model.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>

#include "svo/Data_EuRoc.h"
#include "../src/IMU/imudata.h"
#include "../src/IMU/IMUPreintegrator.h"
#include "../src/IMU/NavState.h"
#include <svo/vio_init.h>

namespace svo {

/// Monocular Visual Odometry Pipeline as described in the SVO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
//  FrameHandlerMono(svo::AbstractCamera* cam ,EuRocData *dataset);
  FrameHandlerMono(svo::PinholeCamera* cam, EuRocData *dadaset);

  virtual ~FrameHandlerMono();

  void Debug_show_img();   // used to debug reproject point

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }
  const vector< pair<FramePtr,size_t> > overlap_kfs(){return overlap_kfs_;}

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }
  const vector<Vector3d>& initFeatureTrackType() const { return klt_homography_init_.fts_type_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  EuRocData* getParam(){return param_;}

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

  ///-------------gzh: add imu part------------------
  void addImu(const IMUData& imu);
  ///-end--

  ///-------------gzh: add vio_init part-------------
  bool vioInitFinish;
  bool imu_update_flag;  // imuState.p.v.R (updated by vision after initialization )can be used to compute prior
  bool getPermission_Read_kf() { return permission_read_kf_;}
  bool getPermission_Update_kf(){ return permission_update_kf_;}
  bool getPermission_ProcessFrame(){ return permission_Process_frame_;}

  void setPermission_Read_kf ( bool permission ){ permission_read_kf_ = permission; }
  void setPermission_Update_kf(bool permission){ permission_update_kf_ = permission;}
  void setPermission_ProcessFrame(bool permission){ permission_Process_frame_ = permission;}

  void setGravityVec(Vector3d gravity){ param_->gravity = gravity;}
  Vector3d getGravityVec(){ return param_->gravity;}

  void setMapScale(double scale) {if(scale > 0) map_scale_ = scale;}  //用不着这个
  double getMapScale(){ return map_scale_; }
  ///-end--

  ///-------------gzh: add imu_prior part-------------
  double priorcount;
  double priorImuWight;
  Vector3d priorImuPos;
  Eigen::Matrix3d priorImuRot;
  Vector3d last_frame_delta_p ;

  double imu_wight_pos_error_sum_;
  double imu_pos_error_sum_;
  double static_pos_error_sum_;
  double uniform_pos_error_sum_;

  int image_id;

  void set_Image_id(int id){ image_id = id;}
  int  get_Image_id(){ return image_id;}
  ///-end--
  ///-------------gzh: add window BA part-------------
  size_t window_size = 10;
  list<FramePtr> win_kfs;
//  int kf_cnt;
//  vector<FramePtr> window_kfs;
//  int window_kf_id = 0;

  void slideWindow();

  ///-end--

protected:
  //svo::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  svo::PinholeCamera* cam_;
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  FramePtr last_kf_;      // hyj: used to last_kf_ to judge the view changes, add new keyframe
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?

  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.
  EuRocData*   param_;                          //!< gzh add: datasets  param_


  ///-------------gzh: add imu part----------------
  enum IMU_STATE
  {
      NO_FRAME,
      NEW_FRAME,
      LAST_FRAME
  };
  IMU_STATE imu_state_;
  IMUPreintegrator imuPreint;
  std::vector<IMUData> m_IMUData;    // gzh :imudata  (t is delta t)
  ///-end--

  ///-------------gzh: add vio_init part-------------
  bool permission_read_kf_;
  bool permission_update_kf_;
  bool permission_Process_frame_;

  double map_scale_;
  ///-end--


  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);

};

} // namespace svo

#endif // SVO_FRAME_HANDLER_H_
