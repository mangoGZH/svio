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

#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
//#include <svo/sparse_img_align.h>
#include <svo/sparse_align.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

//#include "feature_based.h"
//#define DEGUBSHOW

namespace svo {

void FrameHandlerMono::Debug_show_img()
{
    /* debug lost feature */
    cv::Mat img_new = new_frame_->img_pyr_[0].clone();
    std::cout<< "debug show, new frame id : "<< new_frame_->id_ <<"last frame id: "<< last_frame_->id_ <<std::endl;
    for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    {
      if((*it)->type == Feature::EDGELET)
        cv::line(img_new,
                 cv::Point2f((*it)->px[0]+3*(*it)->grad[1], (*it)->px[1]-3*(*it)->grad[0]),
                 cv::Point2f((*it)->px[0]-3*(*it)->grad[1], (*it)->px[1]+3*(*it)->grad[0]),
                 cv::Scalar(0,0,255), 2);
      else
        cv::rectangle(img_new,
                      cv::Point2f((*it)->px[0]-2, (*it)->px[1]-2),
                      cv::Point2f((*it)->px[0]+2, (*it)->px[1]+2),
                      cv::Scalar(0,255,0), CV_FILLED);
    }
    cv::imshow("new_frame",img_new);

    cv::Mat img_last = last_frame_->img_pyr_[0].clone();
    for(Features::iterator it=last_frame_->fts_.begin(); it!=last_frame_->fts_.end(); ++it)
    {
      if((*it)->type == Feature::EDGELET)
        cv::line(img_last,
                 cv::Point2f((*it)->px[0]+3*(*it)->grad[1], (*it)->px[1]-3*(*it)->grad[0]),
                 cv::Point2f((*it)->px[0]-3*(*it)->grad[1], (*it)->px[1]+3*(*it)->grad[0]),
                 cv::Scalar(0,0,255), 2);
      else
        cv::rectangle(img_last,
                      cv::Point2f((*it)->px[0]-2, (*it)->px[1]-2),
                      cv::Point2f((*it)->px[0]+2, (*it)->px[1]+2),
                      cv::Scalar(0,255,0), CV_FILLED);
    }
    cv::imshow("img_last",img_last);

    std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i)
    {
        FramePtr fptr = i.first;
        std::string s = std::to_string( fptr->id_);
        //cv::Mat img = fptr->img_pyr_[0].clone();
        cv::Mat img = fptr->img_pyr_[0].clone();
        for(Features::iterator it = fptr->fts_.begin(); it != fptr->fts_.end(); ++it)
        {
            if( (*it)->type == Feature::EDGELET )
                cv::line(img,
                         cv::Point2f((*it)->px[0]+3*(*it)->grad[1], (*it)->px[1]-3*(*it)->grad[0]),
                         cv::Point2f((*it)->px[0]-3*(*it)->grad[1], (*it)->px[1]+3*(*it)->grad[0]),
                         cv::Scalar(0,0,255), 2);
              else
                cv::rectangle(img,
                              cv::Point2f((*it)->px[0]-2, (*it)->px[1]-2),
                              cv::Point2f((*it)->px[0]+2, (*it)->px[1]+2),
                              cv::Scalar(0,255,0), CV_FILLED);
        }
        imshow(s,img);
    }
    );
   // debug_img_  record the  failure matching feature
    /*
    std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i)
    {
        FramePtr fptr = i.first;
        std::string s = "debug_"+std::to_string( fptr->id_);
        imshow(s,fptr->debug_img_);
    });
    */
    cv::waitKey(0);
}

FrameHandlerMono::FrameHandlerMono(svo::AbstractCamera* cam ,EuRocData* dataset ) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL),
  param_(dataset),
  imu_state_(NO_FRAME)
{
    permission_read_kf_ = true;
    permission_update_kf_ = true;
    permission_Process_frame_ = true;
    vioInitFinish = false;
    //
    initialize();
}

void FrameHandlerMono::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

  feature_detection::DetectorPtr edge_detector(
      new feature_detection::EdgeDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));

  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector,edge_detector ,depth_filter_cb);
  depth_filter_->startThread();
}

FrameHandlerMono::~FrameHandlerMono()
{
  delete depth_filter_;
}

void FrameHandlerMono::addImage(const cv::Mat& img, const double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  //setPermission_Update_kf(false);

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  SVO_STOP_TIMER("pyramid_creation");

#ifdef USE_IMU
  imu_state_ = NEW_FRAME;

  if(stage_ != STAGE_FIRST_FRAME && last_frame_){
      //get imuprent
      new_frame_->SetInitialNavStateAndBias(last_frame_->imuState);
      new_frame_->imuPreint = imuPreint;

      if(vioInitFinish){
          //imu preinit
          //1. provide prior estimate for visual motion
          new_frame_->UpdateNavState( last_frame_->imuState, imuPreint, getGravityVec());
      }
  }
#endif

  // process frame
  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
    res = processFrame();
  else if(stage_ == STAGE_SECOND_FRAME)
    res = processSecondFrame();
  else if(stage_ == STAGE_FIRST_FRAME)
    res = processFirstFrame();
  else if(stage_ == STAGE_RELOCALIZING)
    res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()),
                          map_.getClosestKeyframe(last_frame_));


#ifdef USE_IMU
    ///use vision pose to update the imu pose

    if(vioInitFinish){
//        if( res == RESULT_FAILURE || tracking_quality_ == TRACKING_INSUFFICIENT)
//            priorImuWight=1;
        //TODO: use imu weighted prior
        // How to update the velocity in per-frame ,when relocalization process
        //2. get lastframe velocity from last_frame and cur_frame
        last_frame_->GetpreVelFromeV(new_frame_, getGravityVec(), getParam()->cam_params[0].T_BC, getMapScale());
        //3. compute cur_frame velocity by preinit imu_data from last_frame velocity
        new_frame_->UpdateNavstateFromV(last_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);

    }
#endif
  // debuge
{
    cv::Mat img_new = new_frame_->img_pyr_[0].clone();
    if(img_new.channels()<3) //this should be always true
        cvtColor(img_new,img_new,CV_GRAY2BGR);

  if(stage_ == STAGE_SECOND_FRAME) {
      const vector<cv::Point2f>& px_ref(initFeatureTrackRefPx());
      const vector<cv::Point2f>& px_cur(initFeatureTrackCurPx());
      const vector<Vector3d>& fts_type(initFeatureTrackType());
      vector<Vector3d>::const_iterator it_type = fts_type.begin();
      for(vector<cv::Point2f>::const_iterator it_ref=px_ref.begin(), it_cur=px_cur.begin();it_ref != px_ref.end(); ++it_type,++it_ref, ++it_cur)
       {

        if( (*it_type)[2] == 1)
           cv::line(img_new,
                   cv::Point2f(it_cur->x, it_cur->y),
                   cv::Point2f(it_ref->x, it_ref->y), cv::Scalar(0,0,255), 2);
      else
        cv::line(img_new,
                 cv::Point2f(it_cur->x, it_cur->y),
                 cv::Point2f(it_ref->x, it_ref->y), cv::Scalar(0,255,0), 2);
      }
  }
  else {
      cv::Scalar c;
      if(new_frame_->isKeyframe())
        c = cv::Scalar (255,0,0);
      else
        c = cv::Scalar (0,255,0);

      for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
      {
        if((*it)->type == Feature::EDGELET)
          cv::line(img_new,
                   cv::Point2f((*it)->px[0]+3*(*it)->grad[1], (*it)->px[1]-3*(*it)->grad[0]),
                   cv::Point2f((*it)->px[0]-3*(*it)->grad[1], (*it)->px[1]+3*(*it)->grad[0]),
                   cv::Scalar(0,0,255), 2);
        else
          cv::rectangle(img_new,
                        cv::Point2f((*it)->px[0]-2, (*it)->px[1]-2),
                        cv::Point2f((*it)->px[0]+2, (*it)->px[1]+2),
                        c);
      }
  }

  cv::imshow("new_frame",img_new);
  cv::waitKey(1);
  }  // end debuge

  // set last frame
  last_frame_ = new_frame_;
  if(new_frame_->isKeyframe())
      last_kf_ = new_frame_;
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());

  setPermission_Read_kf(true);
  //setPermission_Update_kf(true);
}

FrameHandlerMono::UpdateResult FrameHandlerMono::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  if(klt_homography_init_.addFirstFrame(new_frame_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;

  setPermission_Read_kf(false);
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);

  if (new_frame_->isKeyframe()) {
      new_frame_->kfIMUData.assign(m_IMUData.begin(), m_IMUData.end());
      new_frame_->ComputePreInt();
      // cout<<"new_frame_->imuPreint.getDeltaTime()"<< new_frame_->imuPreint.getDeltaTime()<<endl;
      // cout<<"new_frame_->kfIMUData.size()　：　" <<new_frame_->kfIMUData.size()<<endl;
      m_IMUData.clear();
  }

  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processSecondFrame()
{
  initialization::InitResult res = klt_homography_init_.addSecondFrame(new_frame_);
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  setPermission_Read_kf(false);
  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add frame to map
  map_.addKeyframe(new_frame_);
  if (new_frame_->isKeyframe()) {
      new_frame_->kfIMUData.assign(m_IMUData.begin(), m_IMUData.end());
      new_frame_->ComputePreInt();
      //cout<<"new_frame_->imuPreint.getDeltaTime()"<< new_frame_->imuPreint.getDeltaTime()<<endl;
      // cout<<"new_frame_->kfIMUData.size()　：　" <<new_frame_->kfIMUData.size()<<endl;
      m_IMUData.clear();
  }

  stage_ = STAGE_DEFAULT_FRAME;
  klt_homography_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMono::processFrame()
{
    while( !getPermission_ProcessFrame()){ // 初始化位姿更新过程中，不能进行新帧处理
        static int wait = 0;
        cout<<"waitting for TryVioInit get Permission ProcessFrame"<< wait++ <<endl;
        usleep(500);
    }

    // Set initial pose
    new_frame_->T_f_w_ = last_frame_->T_f_w_;
    //new_frame_->T_f_w_ = UseimuPrior;TODO use imu prior

  // sparse image align
  SVO_START_TIMER("sparse_img_align");
  /*
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
*/
  //设置稀疏对齐参数                // klt_max_level(4),  klt_min_level(2),   迭代次数  display
  SparseAlign img_align(svo::Config::kltMaxLevel(), svo::Config::kltMinLevel(),30, false, false);
  //开始稀疏对齐                          //优化帧间位姿  optimize(T_cur_from_ref);
  size_t img_align_n_tracked =img_align.run(last_frame_, new_frame_);

  SVO_STOP_TIMER("sparse_img_align");
  SVO_LOG(img_align_n_tracked);
  SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;  // repr_n_new_references为reprojectCell成功次数
  const size_t repr_n_mps = reprojector_.n_trials_;

  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);

  if(repr_n_new_references < Config::qualityMinFts())
  {
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    std::cout<< "\033[1;31m"<<" Not enough matched featrues.  Reproject process failure!"<<" \033[0m" <<std::endl;
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;

    /*debug*/
#ifdef DEGUBSHOW
    Debug_show_img();
#endif
    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final<<"\t delete edges: ="<<new_frame_->fts_.size() - sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final,  reprojector_.n_matches_);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps

    //Debug
#ifdef DEGUBSHOW
    Debug_show_img();
#endif
    return RESULT_FAILURE;
  }

///------ ordinary frame process over ,follwing is keyframe process ---------
   setPermission_Read_kf(false);
// select keyframe   //TODO :gzh :don't use depth_mean to judge keyframe( need better judgement)
  double depth_mean, depth_min; ///当前帧 特征点深度的均值 & 深度的最小值　
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
      (*it)->point->addFrameRef(*it);  //  loop , add a link between feature(in new_frame) to  map point, especially, for new candidates point, Here, we link they to feature in new_frame
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // add keyframe to map
  map_.addKeyframe(new_frame_);
  if (new_frame_->isKeyframe()) {
      new_frame_->kfIMUData.assign(m_IMUData.begin(), m_IMUData.end());
      new_frame_->ComputePreInt();
      //cout<<"new_frame_->imuPreint.getDeltaTime()"<< new_frame_->imuPreint.getDeltaTime()<<endl;
      // cout<<"new_frame_->kfIMUData.size()　：　" <<new_frame_->kfIMUData.size()<<endl;
      m_IMUData.clear();
  }

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

//    last_frame_->GetpreVelFromeV(new_frame_, getGravityVec(), getParam()->cam_params[0].T_BC, getMapScale());
//    new_frame_->UpdateNavstateFromV(last_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);

//setPermission_Read_kf(false);
  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min);

  // add former keyframe to update
  //setCoreKfs(3);
  size_t a = 2; // 3
  size_t n = std::min( a, overlap_kfs_.size()-1);
  auto  it_frame=overlap_kfs_.begin();
  for(size_t i = 0; i<n ; ++i, ++it_frame)
  {
      depth_filter_->addFrame(it_frame->first);
  }

  // if limited number of keyframes, remove the one furthest apart
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  return RESULT_IS_KEYFRAME;
}

FrameHandlerMono::UpdateResult FrameHandlerMono::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_keyframe)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_keyframe == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe.");
    return RESULT_FAILURE;
  }
  /* feature based tracker */
  /*
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  Feature_based_track fb_tracker(feature_detector);
  fb_tracker.track(ref_keyframe,new_frame_);
  */
  // end feature based tracker

  /*
  SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                           30, SparseImgAlign::GaussNewton, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);
  */

  SparseAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),30, false, false);
  size_t img_align_n_tracked = img_align.run(ref_keyframe, new_frame_);

  if(img_align_n_tracked > 30)
  {
    SE3 T_f_w_last = last_frame_->T_f_w_;
    last_frame_ = ref_keyframe;
    FrameHandlerMono::UpdateResult res = processFrame();
    if(res != RESULT_FAILURE)
    {
      stage_ = STAGE_DEFAULT_FRAME;
      SVO_INFO_STREAM("Relocalization successful.");
    }
    else
      new_frame_->T_f_w_ = T_f_w_last; // reset to last well localized pose
    return res;
  }
  return RESULT_FAILURE;
}

bool FrameHandlerMono::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMono::addImu( const IMUData &imu ){ //add imu every frame

    if (imu_state_ == NO_FRAME)
        return;
    if (imu_state_ == NEW_FRAME) {
        imuPreint.reset();
        imu_state_ = LAST_FRAME;
    }

    m_IMUData.push_back(imu);  // (t is delta t)
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;

    if (!last_frame_) {
        //prior bias form okvis(maybe should add this prior in first frame? yes)
        bg.setConstant(0.00);
        ba.setConstant(0.00);
        last_frame_->imuState.Set_BiasGyr(bg);
        last_frame_->imuState.Set_BiasAcc(ba);

    } else {
        bg = last_frame_->imuState.Get_BiasGyr();
        ba = last_frame_->imuState.Get_BiasAcc();
    }
//     cout<<"ba:"<<ba<<endl;
//     cout<<"bg:"<<bg<<endl;
    imuPreint.update(imu._g - bg, imu._a - ba, imu._t);

//    cout<<"\t_delta_P:"<<imuPreint.getDeltaP()<<endl;
//    cout<<"\t_delta_V:"<<imuPreint.getDeltaV()<<endl;
//    cout<<"\t_delta_R:"<<imuPreint.getDeltaR()<<endl;
//    cout<<"\t_delta_t:"<<imuPreint.getDeltaTime()<<endl;
}

void FrameHandlerMono::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMono::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMono::needNewKf(double scene_depth_mean)
{
  vector< double> pixel_dist;
   int cnt=0;

   ///condition1 :上一关键帧的特征点投影到当前帧下的个数>３０ -> goto next condition2
   for(auto it = last_kf_->fts_.begin(), it_end= last_kf_->fts_.end(); it != it_end; ++it)
   {
     // check if the feature has a mappoint assigned
     if((*it)->point == NULL)
       continue;

     Vector2d px1(new_frame_->w2c((*it)->point->pos_));
     Vector2d px2(last_kf_->w2c((*it)->point->pos_));
     Vector2d temp = px1-px2;          //同一地图点在上一关键帧与当前帧投影的像素距离　
     pixel_dist.push_back(temp.norm());
     cnt++;
     if(cnt > Config::KFReprojCNT()) break;

   }
   ///condition2 :  new_frame_ last_kf_共视点的投影像素距离＞４０
  double d = getMedian(pixel_dist);
  if(d > Config::KFMaxPixelDist())   //40
    return true;

  ///condition3 :使用了平移量 确定新的关键
  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
  //TODO:　添加　用旋转量作为选择关键帧的依据  condition4 :使用了旋转量作为确定新的关键帧的策略，
}

void FrameHandlerMono::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){core_kfs_.insert(i.first);});
}

} // namespace svo
