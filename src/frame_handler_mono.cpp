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

FrameHandlerMono::FrameHandlerMono(svo::PinholeCamera* cam, EuRocData *dadaset):
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(nullptr),
  param_(dadaset),
  imu_state_(NO_FRAME)
{
    permission_read_kf_ = true;
    permission_update_kf_ = true;
    permission_Process_frame_ = true;
    vioInitFinish = false;
    imu_update_flag = false;

    priorcount = 0;
    priorImuWight = -1.0;
    imu_wight_pos_error_sum_=0;
    imu_pos_error_sum_=0;
    static_pos_error_sum_=0;
    uniform_pos_error_sum_ =0;

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

  setPermission_Update_kf(false);
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
        //TODO: use imu weighted prior
        // How to update the velocity in per-frame ,when relocalization process
        if( res == RESULT_FAILURE || tracking_quality_ == TRACKING_INSUFFICIENT)
            priorImuWight=1;
        else{
            if (priorImuWight >= 0) {
                double error_imu = (new_frame_->pos() - priorImuPos).norm();
                double error_last = (new_frame_->pos() - last_frame_->pos()).norm();

                //weight comptue method1
                priorImuWight = error_last * error_last / (error_imu * error_imu + error_last * error_last);
                //weight comptue method2
                //priorImuWight = error_last / (error_imu + error_last);
            }
        //2. get lastframe velocity from last_frame and cur_frame
        last_frame_->GetpreVelFromeV(new_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);
        //3. compute cur_frame velocity by preinit imu_data from last_frame velocity
        new_frame_->UpdateNavstateFromV(last_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);
        // only set once ,ensure the imu (p,v,r) effective
        imu_update_flag = true;
        }
    }
#endif
  // debuge
    {
        cv::Mat img_new = new_frame_->img_pyr_[0].clone();
        if (img_new.channels() < 3) //this should be always true
            cvtColor(img_new, img_new, CV_GRAY2BGR);   //将灰度图转化为BGR彩色

        if (stage_ == STAGE_SECOND_FRAME) {             //SECOND_FRAME
            const vector<cv::Point2f> &px_ref(initFeatureTrackRefPx());
            const vector<cv::Point2f> &px_cur(initFeatureTrackCurPx());
            const vector<Vector3d> &fts_type(initFeatureTrackType());
            vector<Vector3d>::const_iterator it_type = fts_type.begin();
            for (vector<cv::Point2f>::const_iterator it_ref = px_ref.begin(), it_cur = px_cur.begin();
                 it_ref != px_ref.end(); ++it_type, ++it_ref, ++it_cur) {

                if ((*it_type)[2] == 1)  // ftr->type == Feature::EDGELET, fts_type[2] = 1;
                    cv::line(img_new,
                             cv::Point2f(it_cur->x, it_cur->y),
                             cv::Point2f(it_ref->x, it_ref->y), cv::Scalar(0, 0, 255), 2);
                else                       // ftr->type == Feature::CORNER, fts_type[2] = 0;
                    cv::line(img_new,
                             cv::Point2f(it_cur->x, it_cur->y),
                             cv::Point2f(it_ref->x, it_ref->y), cv::Scalar(0, 255, 0), 2);
            }
        } else {
            cv::Scalar c;
            if (new_frame_->isKeyframe())
                c = cv::Scalar(255, 0, 0);
            else
                c = cv::Scalar(0, 255, 0);

            for (Features::iterator it = new_frame_->fts_.begin(); it != new_frame_->fts_.end(); ++it) {
                if ((*it)->type == Feature::EDGELET)
                    cv::line(img_new,
                             cv::Point2f((*it)->px[0] + 3 * (*it)->grad[1], (*it)->px[1] - 3 * (*it)->grad[0]),
                             cv::Point2f((*it)->px[0] - 3 * (*it)->grad[1], (*it)->px[1] + 3 * (*it)->grad[0]),
                             cv::Scalar(0, 0, 255), 2);
                else
                    cv::rectangle(img_new,
                                  cv::Point2f((*it)->px[0] - 2, (*it)->px[1] - 2),
                                  cv::Point2f((*it)->px[0] + 2, (*it)->px[1] + 2),
                                  c);
            }
        }

        cv::imshow("new_frame", img_new);
        cv::waitKey(1);
    }  // end debuge

    //get delta_p between cur_frame and last_frame, use to compute uniform_prior
    if(last_frame_ && vioInitFinish ){         // if has last_frame
        last_frame_delta_p = new_frame_->pos()-last_frame_->pos();
//        cout<<"last_frame_delta_p\n"<<last_frame_delta_p<<endl;
    }

    // set last frame
    last_frame_ = new_frame_;
    if(new_frame_->isKeyframe())
        last_kf_ = new_frame_;
    new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());

  setPermission_Read_kf(true);
  setPermission_Update_kf(true);
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
    //TODO use imu prior  new_frame_->T_f_w_ = UseimuPrior;
    if(Config::UseImuPrior()) {

        if (vioInitFinish && imu_update_flag) {
            if(priorImuWight < 0.0 )
                priorImuWight = 0.0;
            priorcount++;
            // update visual new_frame pose from [ imustate + imupreint ]
            new_frame_->UpdatePoseFromNS(getParam()->cam_params[0].T_BC);
            priorImuPos = new_frame_->pos();
            priorImuRot = new_frame_->rot();

            ///prior method test code
            //get groundtruth position:  groundtruth is align to body_frame
            Vector3d truth_translation =
                    (param_->getTruthTranslation(get_Image_id())- param_->getTruthTranslation(get_Image_id()-1));

            Vector3d imu_translation =
                    (new_frame_->pos() - last_frame_->pos());

//            Matrix3d R_g0_c0 = Matrix3d::Identity();
            cv::Mat Tbc = param_->cam_params[0].T_BC;  // trans c to b_frame
            cv::Mat Rbc = Tbc.rowRange(0, 3).colRange(0, 3);
            cv::Mat pbc = Tbc.rowRange(0, 3).col(3);
            Matrix3d R_bc = Converter::toMatrix3d(Rbc);
            Vector3d p_bc = Converter::toVector3d(pbc);

            Matrix3d R_truth_last = param_->getTrueQuaternion(get_Image_id() -1).normalized().toRotationMatrix();//get q_truth_last
            Matrix3d R_truth_start = param_->getTrueQuaternion(Config::StartFrame()).normalized().toRotationMatrix();
            Matrix3d R_truth_cur = param_->getTrueQuaternion(get_Image_id()).normalized().toRotationMatrix();
            Matrix3d cur_rot_start_cur = R_truth_start.inverse() * R_truth_cur;          //cur_rot_w_b:  R_truth from cur to start

            //rotation form ground_w to c_w frame
            Matrix3d R_cw_gw = (R_truth_start * R_bc).transpose();
            //delta_translation_cw  is the same as delta_translation_
            Vector3d delta_translation_cw = R_cw_gw* truth_translation;
            Vector3d delta_translation_ = R_truth_last.transpose() * truth_translation;      // R_g_g0.inv()* delta_p[g0] = dalta_p[g]

            //priorPos_w_b = lastpos_w_b + lastrot_w_b*dalta_p[g]
            Vector3d priorPos_truth_b = last_frame_->imuState.Get_P()
                                      + last_frame_->imuState.Get_RotMatrix() *delta_translation_;

            //对比真实旋转量与imu预积分旋转量的差异  //truth_rot= R_last_w * R_w_cur;
            Matrix3d truth_rot = R_truth_last.transpose()*R_truth_cur;   //当前帧相对于上一帧的相对旋转　（真值）
            Matrix3d imu_rot = new_frame_->imuPreint.getDeltaR();

            //使用当前与上一帧间的真值旋转量　位移增量加到ｂ,再加上ｂｃ之间的位移，得到ｃ相对于世界系的位置　// priorPos_w_c = priorPos_w_b + cur_rot_w_b * p_b_c
            Vector3d priorPos_truth_w = (last_frame_->imuState.Get_RotMatrix() * truth_rot )* p_bc + priorPos_truth_b;
            //未使用真值旋转量 位移增量加到ｃ系再转到世界系
            //Vector3d priorPos_truth_cw = last_frame_->pos() + last_frame_->rot() * R_bc.transpose() * delta_translation_;
            //由于得到了在相机世界系下的位移增量　delta_translation_cw,可以直接叠加到相机在世界系的位置上
            Vector3d priorPos_truth_cw = last_frame_->pos() + delta_translation_cw;
            Vector3d priorPos_unifor  = last_frame_->pos() + last_frame_delta_p;

//            cout<<"priorPos_truth_w is \n"<< priorPos_truth_w <<endl;
//            cout<<"priorImuPos\n"<< priorImuPos<<endl;
//            cout<<"priorPos_unifor is \n"<< priorPos_unifor <<endl;

            double static_pos_error = (last_frame_->pos() - priorPos_truth_w ).norm();
            double imu_pos_error = (priorImuPos - priorPos_truth_w).norm();
                                                           // v_wc =R_bc* v_wb
//            double uniform_pos_error1 = (last_frame_->pos() + R_bc * last_frame_->imuState.Get_V()*0.05 -priorPos_truth_w ).norm();
            double uniform_pos_error = (priorPos_unifor - priorPos_truth_w).norm();

            double weight_prior_error = ((new_frame_->pos() - last_frame_->pos()) * priorImuWight - delta_translation_cw).norm();

            //加权模型
            imu_wight_pos_error_sum_ += weight_prior_error;
            double imu_wight_error_dist_average = imu_wight_pos_error_sum_ / priorcount;
            //静止模型
            static_pos_error_sum_ += static_pos_error;
            double static_pos_error_average = static_pos_error_sum_ / priorcount;
            //ｉｍｕ先验模型
            imu_pos_error_sum_ += imu_pos_error;
            double imu_pos_error_average = imu_pos_error_sum_ / priorcount;
            //匀速模型
            uniform_pos_error_sum_ += uniform_pos_error;
            double uniform_pos_error_average = uniform_pos_error_sum_ / priorcount;

            //dubug
            cout << "[test UseIMUPrior]-----id: "<< get_Image_id() <<"    priorImuWight: "<<priorImuWight<< endl;
            cout<< "priorcount : \n"<< priorcount<<endl;
            cout<< "static_pos_error : \n"<< static_pos_error<<endl;
            cout<< "imu_pos_error : \n"<< imu_pos_error<<endl;
            cout<< "uniform_pos_error : \n"<< uniform_pos_error<<endl;
            cout<< "weight_prior_error : \n"<< weight_prior_error<<endl;

//            cout<< "static_pos_error_average : \n"<< static_pos_error_average<<endl;
//            cout<< "imu_pos_error_average : \n"<< imu_pos_error_average<<endl;
//            cout<< "uniform_pos_error_average : \n"<< uniform_pos_error_average<<endl;
//            cout<< "imu_wight_error_dist_average : \n"<< imu_wight_error_dist_average<<endl;

            //----------FILEOUTPUT: prior_error data(compare with the groundtruth)-----------
            static bool fileopened = false;
            static ofstream prior_error;
            string  filepath = "/home/gzh/SVIO_rebuild/svio_outfile/";
            if(!fileopened)
            {
                prior_error.open(filepath+"prior_error.txt");
                if( prior_error.is_open() )
                    fileopened = true;
                else
                {
                    cerr<<"fileopened open error in prior_error file"<<endl;
                    fileopened = false;
                }
                prior_error<<std::fixed<<std::setprecision(6);
            }
            prior_error << new_frame_->timestamp_<<" "
                        << new_frame_->id_<<" "
                        << static_pos_error <<" "
                        << imu_pos_error <<" "                       //imu_error_dist_cur
                        << uniform_pos_error <<" "
                        << weight_prior_error <<" "                  //imu_wight_error_dist_cur
                        << endl;        //uniform_prior_error
            //-------------------------------------------
            //weight_prior
            new_frame_->T_f_w_.inverse().translation() = new_frame_->pos()*priorImuWight +
                                                         last_frame_->pos()*(1 - priorImuWight);
            new_frame_->T_f_w_.inverse().rotation_matrix() = new_frame_->rot() * priorImuWight +
                                                             last_frame_->rot() * (1 - priorImuWight);
//            cout<<"use weight_prior"<<endl;

            //uniform_prior
//            new_frame_->T_f_w_.inverse().translation() =  priorPos_unifor;
//            cout<<"use uniform_prior"<<endl;

            //truth_prior
            //new_frame_->T_f_w_.inverse().translation() = priorPos_truth_w;
//            cout<<"use truth_prior"<<endl;

            //imu_prior

        }else{
            new_frame_->T_f_w_ = last_frame_->T_f_w_;
            //cout<<"new_frame_->T_f_w_ = last_frame_->T_f_w_;"<<endl;
        }
    }else{
        //设置初始位姿，将上帧（last_frame_）的变换矩阵（T_f_w_）赋给当前帧}
        new_frame_->T_f_w_ = last_frame_->T_f_w_;
        //cout<<"new_frame_->T_f_w_ = last_frame_->T_f_w_;"<<endl;
    }

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

  //记录上一帧的match,用来比较上一帧和当前帧匹配点数量变化大小－－用于关键帧判断
  const size_t last_matches = reprojector_.n_matches_;

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;  // repr_n_new_references为reprojectCell成功次数
  const size_t repr_n_mps = reprojector_.n_trials_;

  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);

  int match_decrease = repr_n_new_references - last_matches;
  cout<<"match_decrease:\t"<<match_decrease<<"\n"<<endl;

  // 新跟踪的特征点投影到３Ｄ中，再投影回关键帧上进行匹配
  // 有时会因为观测值较少，深度没有收敛，与环境投影匹配阶段会丢失很多特征点
  // 造成重投影失败，运动跟踪重定位失败的结果：
  if (match_decrease < -50) //若当前帧跟踪的特征点数比上一帧跟踪的特征点数　少50　
  {                         //此处使用之前IMU的预积分值赋值给 位姿估计
      new_frame_->T_f_w_.inverse().translation() = priorImuPos;
      new_frame_->T_f_w_.inverse().rotation_matrix() = priorImuRot;
      tracking_quality_ = TRACKING_INSUFFICIENT;
      std::cout<< "\033[1;31m"<<" matched featrues decrease more than 50 points.  TRACKING_INSUFFICIENT !\n"<<" \033[0m" <<std::endl;
  }

//  if(repr_n_new_references < Config::qualityMinFts())
//  {
//    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
//    std::cout<< "\033[1;31m"<<" Not enough matched featrues.  Reproject process failure!"<<" \033[0m" <<std::endl;
//    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
//    tracking_quality_ = TRACKING_INSUFFICIENT;
//
//    /*debug*/
//#ifdef DEGUBSHOW
//    Debug_show_img();
//#endif
//    return RESULT_FAILURE;
//  }

  // pose optimization T_new
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final
                          <<"\t delete edges: ="<<new_frame_->fts_.size() - sfba_n_edges_final);

  if(sfba_n_edges_final < 20)
    return RESULT_FAILURE;

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  setTrackingQuality(sfba_n_edges_final, reprojector_.n_matches_);

//  if(tracking_quality_ == TRACKING_INSUFFICIENT)
//  {
//    new_frame_->T_f_w_ = last_frame_->T_f_w_; // reset to avoid crazy pose jumps
//
//    //Debug
//#ifdef DEGUBSHOW
//    Debug_show_img();
//#endif
//    return RESULT_FAILURE;
//  }

///------ ordinary frame process over ,follwing is keyframe process ---------
//   setPermission_Read_kf(false);

// select keyframe   //TODO :gzh :don't use depth_mean to judge keyframe( need better judgement)
  double depth_mean, depth_min; ///当前帧 特征点深度的均值 & 深度的最小值　
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);

  if(!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD)
  {
    depth_filter_->addFrame(new_frame_);
    return RESULT_NO_KEYFRAME;
  }

  setPermission_Read_kf(false);
  core_kfs_.insert(new_frame_);
  new_frame_->setKeyframe();
  SVO_DEBUG_STREAM("New keyframe selected.");

  // new keyframe selected
  for(Features::iterator it = new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != nullptr)
      (*it)->point->addFrameRef(*it);  //  loop , add a link between feature(in new_frame) to  map point, especially, for new candidates point, Here, we link they to feature in new_frame
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // add keyframe to map
  map_.addKeyframe(new_frame_);
  setCoreKfs(Config::coreNKfs());

  if (new_frame_->isKeyframe()) {
      new_frame_->kfIMUData.assign(m_IMUData.begin(), m_IMUData.end());
      new_frame_->ComputePreInt();
      //cout<<"new_frame_->imuPreint.getDeltaTime()"<< new_frame_->imuPreint.getDeltaTime()<<endl;
      // cout<<"new_frame_->kfIMUData.size()　：　" <<new_frame_->kfIMUData.size()<<endl;
      m_IMUData.clear();
  }

  if(Config::UseImuPrior()) {
      last_frame_->GetpreVelFromeV(new_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);
      new_frame_->UpdateNavstateFromV(last_frame_, getGravityVec(), getParam()->cam_params[0].T_BC);
  }
   // cout<<"before localba---new_frame_->pos()\n"<<new_frame_->pos() <<endl;

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  //test localba time
  clock_t loba_stime,loba_etime;
  double loba_time;

  /// 1.localBA
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    loba_stime = clock();

    setCoreKfs(Config::coreNKfs());      // core_n_kfs(n) = n_closest
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;

    // all core keyframes and all points observed are optimized:
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);

    loba_etime = clock();
    loba_time = (double)(loba_etime - loba_stime)/CLOCKS_PER_SEC;

    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_LOG(loba_time);
    SVO_DEBUG_STREAM(
            "Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
            "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}\t"
            "Time Consumption: "<< loba_time<< "\t");
  }
    //cout<<"before windowba---new_frame_->pos()\n"<<new_frame_->pos() <<endl;

  /// 2.windowBA
  if(vioInitFinish && new_frame_->isKeyframe() && Config::winbaNumIter()> 0) {

      SVO_START_TIMER("window_ba");
      clock_t winba_stime, winba_etime;
      double winba_time;
      winba_stime = clock();  //gzh fixed 2020-1-26

      if(win_kfs.size() < window_size){
          win_kfs.push_back(new_frame_);
// debug
//          for(list<FramePtr>::iterator it =win_kfs.begin();it!=win_kfs.end();++it){
//              Frame *pkf = it->get();
//              cout<<"slideWindow:have kf \n"<<pkf->kf_id_<<endl;}

      }else{//win_kfs.size() == window_size

          //1 margin the oldest_kf,  compute the marginH TODO: margin orgorithm
          //要marg那些不被其他帧观测到的特征点　//ＦＥＪ：固定第一个雅克比
          // marginalization();
          //2 remove the old kf ,which has few overlap features
          slideWindow();
          cout<<"win_kfs.size():\t"<< win_kfs.size()<<endl;

          //3 optimize: window keyframes and  points observed and imu parameters and scale
          size_t winba_n_erredges_init, winba_n_erredges_fin;
          double winba_err_init, winba_err_fin;

          ba::windowBA(new_frame_.get(), &win_kfs, &map_, getGravityVec(), getParam()->cam_params[0].T_BS,
                       winba_n_erredges_init, winba_n_erredges_fin,
                       winba_err_init, winba_err_fin);
//          ba::windowBA(new_frame_.get(), &window_kfs, &map_, getGravityVec(), getParam()->cam_params[0].T_BS,
//                       winba_n_erredges_init, winba_n_erredges_fin,
//                       winba_err_init, winba_err_fin);

          winba_etime = clock();
          winba_time = (double) (winba_etime - winba_stime) / CLOCKS_PER_SEC;

          SVO_STOP_TIMER("window_ba");
          SVO_LOG4(winba_n_erredges_init, winba_n_erredges_fin, winba_err_init, winba_err_fin);
          SVO_LOG(winba_time);
          SVO_DEBUG_STREAM(
                  "Window BA:\t RemovedEdges {" << winba_n_erredges_init << ", " << winba_n_erredges_fin << "} \t "
                  "Error {"<< winba_err_init << ", " << winba_err_fin << "}\t"
                  "Time Consumption:"<< winba_time << "\t");
      }

//      if (kf_cnt != window_size) //10
//      {
//          kf_cnt++;
//          window_kfs.push_back(new_frame_);
//          new_frame_->window_kf_id_ = ++window_kf_id;
//
//      } else {
//          //1 margin the oldest_kf,  compute the marginH TODO: margin orgorithm
//          //要marg那些不被其他帧观测到的特征点　//ＦＥＪ：固定第一个雅克比
//          // marginalization();
//          //2 remove the old kf ,which has few overlap features
//          slideWindow();
//
//          window_kfs.push_back(new_frame_);
//          new_frame_->window_kf_id_ = ++window_kf_id;
//
//          //3 optimize: window keyframes and  points observed and imu parameters and scale
//          winba_stime = clock();  //gzh fixed 2020-1-26
//          size_t winba_n_erredges_init, winba_n_erredges_fin;
//          double winba_err_init, winba_err_fin;
//
//          ba::windowBA(new_frame_.get(), &window_kfs, &map_, getGravityVec(), getParam()->cam_params[0].T_BS,
//                       winba_n_erredges_init, winba_n_erredges_fin,
//                       winba_err_init, winba_err_fin);
//
//          winba_etime = clock();
//          winba_time = (double) (winba_etime - winba_stime) / CLOCKS_PER_SEC;
//
//          SVO_STOP_TIMER("window_ba");
//          SVO_LOG(winba_time);
//          SVO_LOG4(winba_n_erredges_init, winba_n_erredges_fin, winba_err_init, winba_err_fin);
//          SVO_DEBUG_STREAM(
//                  "Window BA:\t RemovedEdges {" << winba_n_erredges_init << ", " << winba_n_erredges_fin << "} \t "
//                  "Error {"<< winba_err_init << ", " << winba_err_fin << "}\t"
//                  "Time Consumption{"<< winba_time << "}\t");
//      }
  }
#endif

//setPermission_Read_kf(false);
  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5 * depth_min);

  // add former keyframe to update
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

//    cout<<"\t_delta_P:\n"<<imuPreint.getDeltaP()<<endl;
//    cout<<"\t_delta_V:\n"<<imuPreint.getDeltaV()<<endl;
//    cout<<"\t_delta_R:\n"<<imuPreint.getDeltaR()<<endl;
//    cout<<"\t_delta_t:\n"<<imuPreint.getDeltaTime()<<endl;
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
   ///condition0  : 若当前帧跟踪的特征点数比上一帧跟踪的特征点数　少30　

   ///condition1 :上一关键帧的特征点投影到当前帧下的个数>３０ -> goto next condition2
   for(auto it = last_kf_->fts_.begin(), it_end= last_kf_->fts_.end(); it != it_end; ++it)
   {
     // check if the feature has a mappoint assigned
     if((*it)->point == nullptr)
       continue;
    //同一地图点point在上一关键帧与当前帧投影的像素距离　
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
    Vector3d lastkf_pos = new_frame_->w2f(last_kf_->pos());
   if (fabs(lastkf_pos.x()) / scene_depth_mean < Config::kfSelectMinDist() * 0.5 && //*1
        fabs(lastkf_pos.y()) / scene_depth_mean < Config::kfSelectMinDist() * 0.8 && //0.8
        fabs(lastkf_pos.z()) / scene_depth_mean < Config::kfSelectMinDist() * 0.5)  //*1.3
   {
        return false;
    }
    return true;
//oriang method
//  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
//  {
//    Vector3d relpos = new_frame_->w2f(it->first->pos());
//    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
//       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
//       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
//      return false;
//  }
//  return true;
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
//void FrameHandlerMono::slideWindow(){
//    if(kf_cnt == window_size){
//        for(vector<FramePtr>::iterator it_kf = window_kfs.begin(); it_kf != window_kfs.end(); ++it_kf){
//            Frame *pkf = it_kf->get();
//            pkf->window_kf_id_ = pkf->window_kf_id_-1;
//        }
//        window_kf_id = window_kfs.back()->window_kf_id_;
//       // window_kfs.front()->window_kf_id_ =NULL;
//        window_kfs.erase(window_kfs.begin());
//    }
//}

void FrameHandlerMono::slideWindow() {

    if(window_size == win_kfs.size()){

        win_kfs.pop_front();
        cout<<"slideWindow pop front: "<<win_kfs.front()->kf_id_<<"win_kfs.size()"<<win_kfs.size()<<endl;
        win_kfs.push_back(new_frame_);
        cout<<"slideWindow push_back: "<<new_frame_->kf_id_<<"win_kfs.size()"<<win_kfs.size()<<endl;
        //debug
//        for(list<FramePtr>::iterator it =win_kfs.begin();it!=win_kfs.end();++it){
//            Frame *pkf = it->get();
//            cout<<"slideWindow:push_back\n"<<pkf->kf_id_<<endl;
//        }
    }
}
} // namespace svo
