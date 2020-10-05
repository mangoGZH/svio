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

#include <stdexcept>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <boost/bind.hpp>

#if __SSE2__
# include <emmintrin.h>
#elif __ARM_NEON__
# include <arm_neon.h>
#endif

namespace svo {

int Frame::frame_counter_ = 0;

//Frame::Frame(svo::AbstractCamera* cam, const cv::Mat& img, double timestamp) :
Frame::Frame(svo::PinholeCamera* cam, const cv::Mat& img, double timestamp) :
    id_(frame_counter_++),
    timestamp_(timestamp),
    cam_(cam),
    key_pts_(5),
    is_keyframe_(false),
    have_initializeSeeds(false),
    v_kf_(nullptr),
    PrevKeyFrame(nullptr),
    v_PVR_(nullptr),   //gzh add 2019-9-23
    v_Bias_(nullptr),
    is_infixed_kf(false),
    is_inWindow_(false),
    winmap_obs_ftr(0)
{
  initFrame(img);
}

Frame::~Frame()
{
  std::for_each(fts_.begin(), fts_.end(), [&](Feature* i){delete i;});
}

void Frame::initFrame(const cv::Mat& img)
{
  // check image
  if(img.empty() || img.type() != CV_8UC1 || img.cols != cam_->width() || img.rows != cam_->height())
    throw std::runtime_error("Frame: provided image has not the same size as the camera model or image is not grayscale");

  // Set keypoints to NULL
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature* ftr){ ftr=NULL; });

  // Build Image Pyramid
  frame_utils::createImgPyramid(img, max(Config::nPyrLevels(), Config::kltMaxLevel()+1), img_pyr_);
}

void Frame::setKeyframe()
{
  is_keyframe_ = true;
  setKeyPoints();
}

void Frame::addFeature(Feature* ftr)
{
  fts_.push_back(ftr);
}

void Frame::setKeyPoints()
{
  for(size_t i = 0; i < 5; ++i)
    if(key_pts_[i] != NULL)
      if(key_pts_[i]->point == NULL)
        key_pts_[i] = NULL;

  std::for_each(fts_.begin(), fts_.end(), [&](Feature* ftr){ if(ftr->point != NULL) checkKeyPoints(ftr); });
}

void Frame::checkKeyPoints(Feature* ftr)
{
  const int cu = cam_->width()/2;
  const int cv = cam_->height()/2;

  // center pixel
  if(key_pts_[0] == NULL)
    key_pts_[0] = ftr;
  else if(std::max(std::fabs(ftr->px[0]-cu), std::fabs(ftr->px[1]-cv))
        < std::max(std::fabs(key_pts_[0]->px[0]-cu), std::fabs(key_pts_[0]->px[1]-cv)))
    key_pts_[0] = ftr;

  if(ftr->px[0] >= cu && ftr->px[1] >= cv)
  {
    if(key_pts_[1] == NULL)
      key_pts_[1] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[1]->px[0]-cu) * (key_pts_[1]->px[1]-cv))
      key_pts_[1] = ftr;
  }
  if(ftr->px[0] >= cu && ftr->px[1] < cv)
  {
    if(key_pts_[2] == NULL)
      key_pts_[2] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          < (key_pts_[2]->px[0]-cu) * (key_pts_[2]->px[1]-cv))
      key_pts_[2] = ftr;
  }
  if(ftr->px[0] < cu && ftr->px[1] < cv)
  {
    if(key_pts_[3] == NULL)
      key_pts_[3] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          > (key_pts_[3]->px[0]-cu) * (key_pts_[3]->px[1]-cv))
      key_pts_[3] = ftr;
  }
  if(ftr->px[0] < cu && ftr->px[1] >= cv)
  {
    if(key_pts_[4] == NULL)
      key_pts_[4] = ftr;
    else if((ftr->px[0]-cu) * (ftr->px[1]-cv)
          < (key_pts_[4]->px[0]-cu) * (key_pts_[4]->px[1]-cv))
      key_pts_[4] = ftr;
  }
}

void Frame::removeKeyPoint(Feature* ftr)
{
  bool found = false;
  std::for_each(key_pts_.begin(), key_pts_.end(), [&](Feature*& i){
    if(i == ftr) {
      i = NULL;
      found = true;
    }
  });
  if(found)
    setKeyPoints();
}

bool Frame::isVisible(const Vector3d& xyz_w) const
{
  Vector3d xyz_f = T_f_w_*xyz_w;
  if(xyz_f.z() < 0.0)
    return false; // point is behind the camera
  Vector2d px = f2c(xyz_f);
  if(px[0] >= 0.0 && px[1] >= 0.0 && px[0] < cam_->width() && px[1] < cam_->height())
    return true;
  return false;
}

void Frame::SetPrevKeyFrame(Frame* pKF) {
    // unique_lock<mutex> lock(mMutexPrevKF);
    PrevKeyFrame = pKF;
}

void Frame::SetNextKeyFrame(Frame* pKF) {
    // unique_lock<mutex> lock(mMutexNextKF);
    NextKeyFrame = pKF;
}

Frame* Frame::GetPrevKeyFrame() {
    //unique_lock<mutex> lock(mMutexPrevKF);
    return PrevKeyFrame;
}

Frame* Frame::GetNextKeyFrame() {
    // unique_lock<mutex> lock(mMutexNextKF);
    return NextKeyFrame;
}

cv::Mat Frame::GetVPoseInverse() { //vTwc:  相机坐标c 到世界坐标w 的变换矩阵   wTc
    cv::Mat vTwc = Converter::toCvMat(T_f_w_.inverse().matrix()); //T_f_w_:从世界坐标到相机坐标的变换矩阵  cTw
    return vTwc;
}

cv::Mat Frame::GetVPos() {         //vPcw:  世界坐标w 到 相机c 的位移
    cv::Mat vPcw = Converter::toCvMat(T_f_w_.translation());      //T_f_w_:从世界坐标到相机坐标的变换矩阵
    return vPcw;
}

void Frame::SetVPose(const cv::Mat &tcw) {

    T_f_w_.translation() = Converter::toVector3d(tcw);

    tcw.copyTo(Tcw.rowRange(0, 3).col(3));
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc * tcw;

    Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc.rowRange(0, 3).col(3));

}
///***************window BA part***********
void Frame::putKFtofixedKF()
{
    is_infixed_kf = true;
}

///***************imu part*****************
void Frame::ComputePreInt() {      //ComputePreInt between key frame
    if (PrevKeyFrame == NULL) {
        return;
    } else {
        kfimuPreint.reset();

        // IMU pre-integration integrates IMU data from last to current, but the bias is from last
        Vector3d bg = PrevKeyFrame->GetNavState().Get_BiasGyr();

        Vector3d ba = PrevKeyFrame->GetNavState().Get_BiasAcc();

        for (size_t i = 0; i < kfIMUData.size(); i++) {
            const IMUData &imu = kfIMUData[i];

            kfimuPreint.update(imu._g - bg, imu._a - ba, imu._t);
        }
    }
}

const NavState& Frame::GetNavState() {
    // unique_lock<mutex> lock(mMutexNavState);
    return imuState;
}

void Frame::SetNavStatePos(const Vector3d &pos) {
    //unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_Pos(pos);
}

void Frame::SetNavStateVel(const Vector3d &vel) {
    //unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_Vel(vel);
}

void Frame::SetNavStateRot(const Matrix3d &rot) {
    //unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_Rot(rot);
}

void Frame::SetNavStateRot(const Sophus::SO3 &rot){
    imuState.Set_Rot(rot);
}

void Frame::SetNavStateBiasGyr(const Vector3d &bg) {
    // unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_BiasGyr(bg);
}

void Frame::SetNavStateBiasAcc(const Vector3d &ba) {
    //unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_BiasAcc(ba);
}

void Frame::SetNavStateDeltaBg(const Vector3d &dbg) {
    // unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_DeltaBiasGyr(dbg);
}

void Frame::SetNavStateDeltaBa(const Vector3d &dba) {
    //unique_lock<mutex> lock(mMutexNavState);
    imuState.Set_DeltaBiasAcc(dba);
}


void Frame::SetInitialNavStateAndBias(const NavState& ns) {
    imuState = ns;
    imuState.Set_BiasGyr(ns.Get_BiasGyr() + ns.Get_dBias_Gyr());
    imuState.Set_BiasAcc(ns.Get_BiasAcc() + ns.Get_dBias_Acc());
    imuState.Set_DeltaBiasGyr(Eigen::Vector3d::Zero());
    imuState.Set_DeltaBiasAcc(Eigen::Vector3d::Zero());
}

//pre-integration in neighbor two frame
void Frame::UpdateNavState(const NavState& lastImuState, const IMUPreintegrator& imupreint, const Eigen::Vector3d& gw) {

    Eigen::Matrix3d dR = imupreint.getDeltaR();//_delta_R;
    Eigen::Vector3d dP = imupreint.getDeltaP();//_delta_P;
    Eigen::Vector3d dV = imupreint.getDeltaV();//_delta_V;
    double dt = imupreint.getDeltaTime();      //_delta_time;

    Eigen::Vector3d Pwbpre = lastImuState.Get_P();
    Eigen::Matrix3d Rwbpre = lastImuState.Get_RotMatrix();
    Eigen::Vector3d Vwbpre = lastImuState.Get_V();

    Eigen::Matrix3d Rwb = Rwbpre * dR;
    Eigen::Vector3d Pwb = Pwbpre + Vwbpre*dt + 0.5 * gw * dt * dt + Rwbpre * dP;
    Eigen::Vector3d Vwb = Vwbpre + gw * dt + Rwbpre * dV;

    // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
    imuState.Set_Pos(Pwb);
    imuState.Set_Vel(Vwb);
    imuState.Set_Rot(Rwb);

    // Test log
    if (imuState.Get_dBias_Gyr().norm()>1e-6 || imuState.Get_dBias_Acc().norm()>1e-6)
        std::cerr << "delta bias in updateNS is not zero" << imuState.Get_dBias_Gyr().transpose()
        << ", " << imuState.Get_dBias_Acc().transpose() << std::endl;
}

///use current visual state set the last imustate(Velocity) in every frame.
void Frame::GetpreVelFromeV( FramePtr curframe, const Vector3d gravity, const cv::Mat &Tbc) {

    // Extrinsics
    cv::Mat Rbc = Tbc.rowRange(0, 3).colRange(0, 3);
    cv::Mat pbc = Tbc.rowRange(0, 3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb * pbc;

    // Position and rotation of visual SLAM
    cv::Mat wPc_last = GetVPoseInverse().rowRange(0, 3).col(3);                   // wPc
    cv::Mat Rwc_last = GetVPoseInverse().rowRange(0, 3).colRange(0, 3);            // Rwc

    // IMU pre-int between last frame ~ current frame
    const IMUPreintegrator &imupreint_last_cur = curframe->imuPreint;

    // Time from this(pKF) to next(pKFnext)
    double dt = imupreint_last_cur.getDeltaTime();
    cv::Mat dp = Converter::toCvMat(imupreint_last_cur.getDeltaP());       // deltaP
    cv::Mat Jpba = Converter::toCvMat(imupreint_last_cur.getJPBiasa());    // J_deltaP_biasa
    Eigen::Vector3d dbiasa_eig = curframe->imuState.Get_BiasAcc();
    cv::Mat biasa_ = Converter::toCvMat(dbiasa_eig);
    cv::Mat gravity_ = Converter::toCvMat(gravity);

    cv::Mat wPc_cur = curframe->GetVPoseInverse().rowRange(0, 3).col(3);           // wPc next
    cv::Mat Rwc_cur = curframe->GetVPoseInverse().rowRange(0, 3).colRange(0, 3);    // Rwc next

    cv::Mat vel = -1. / dt * ((wPc_last - wPc_cur) + (Rwc_last - Rwc_cur) * pcb +
            Rwc_last * Rcb * (dp + Jpba * biasa_) + 0.5 * gravity_ * dt * dt);
    Eigen::Vector3d veleig = Converter::toVector3d(vel);
    SetNavStateVel(veleig);

    //cout<<" SetNavStateVel (use current visual state to compute last v) veleig = "<<veleig<<endl;
}

void Frame::UpdateNavstateFromV(FramePtr prevframe, Vector3d gravity, const cv::Mat& Tbc){

    // Extrinsics
    //cv::Mat Tbc = _vo->getParam()->cam_params[0].T_BC;
    cv::Mat Rbc = Tbc.rowRange(0, 3).colRange(0, 3);
    cv::Mat pbc = Tbc.rowRange(0, 3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb * pbc;

    cv::Mat wPc = GetVPoseInverse().rowRange(0, 3).col(3);             // wPc
    cv::Mat Rwc = GetVPoseInverse().rowRange(0, 3).colRange(0, 3);     // Rwc
    cv::Mat wPb = wPc + Rwc*pcb;

    SetNavStatePos(Converter::toVector3d(wPb));
    SetNavStateRot(Converter::toMatrix3d(Rwc*Rcb));

    Frame *prev_frame = prevframe.get();
    const IMUPreintegrator &imupreint_prev_cur = imuPreint;
    double dt = imupreint_prev_cur.getDeltaTime();
    Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
    Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
    Eigen::Vector3d dbiasa_eig = prev_frame->imuState.Get_BiasAcc();

    Eigen::Vector3d velpre = prev_frame->GetNavState().Get_V();
    Eigen::Matrix3d rotpre = prev_frame->GetNavState().Get_RotMatrix();
    Eigen::Vector3d Pospre =  prev_frame->GetNavState().Get_P();

    Eigen::Vector3d veleig = velpre + gravity * dt + rotpre * (dv + Jvba * dbiasa_eig);
    SetNavStateVel(veleig);
    //cout<<" SetNavStateVel (use last visual state to update current v) veleig = "<<veleig<<endl;
}


Sophus::SE3 Frame::UpdatePoseFromNS(const cv::Mat& Tbc){

    cv::Mat Rbc_ = Tbc.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Pbc_ = Tbc.rowRange(0,3).col(3).clone();

    cv::Mat Rwb_ = Converter::toCvMat(imuState.Get_RotMatrix());
    cv::Mat Pwb_ = Converter::toCvMat(imuState.Get_P());

    cv::Mat Rcw_ = (Rwb_*Rbc_).t();//inv()
    cv::Mat Pwc_ = Rwb_*Pbc_ + Pwb_;
    cv::Mat Pcw_ = -Rcw_*Pwc_;

    T_f_w_ = SE3(Converter::toMatrix3d(Rcw_),
                 Converter::toVector3d(Pcw_));  //gzh fixed 2019-9-23
    return T_f_w_;
}



/// Utility functions for the Frame class
namespace frame_utils {

#ifdef __SSE2__
void halfSampleSSE2(const unsigned char* in, unsigned char* out, int w, int h)
{
  const unsigned long long mask[2] = {0x00FF00FF00FF00FFull, 0x00FF00FF00FF00FFull};
  const unsigned char* nextRow = in + w;
  __m128i m = _mm_loadu_si128((const __m128i*)mask);
  int sw = w >> 4;
  int sh = h >> 1;
  for (int i=0; i<sh; i++)
  {
    for (int j=0; j<sw; j++)
    {
      __m128i here = _mm_load_si128((const __m128i*)in);
      __m128i next = _mm_load_si128((const __m128i*)nextRow);
      here = _mm_avg_epu8(here,next);
      next = _mm_and_si128(_mm_srli_si128(here,1), m);
      here = _mm_and_si128(here,m);
      here = _mm_avg_epu16(here, next);
      _mm_storel_epi64((__m128i*)out, _mm_packus_epi16(here,here));
      in += 16;
      nextRow += 16;
      out += 8;
    }
    in += w;
    nextRow += w;
  }
}
#endif

#ifdef __ARM_NEON__
void halfSampleNEON( const cv::Mat& in, cv::Mat& out )
{
  for( int y = 0; y < in.rows; y += 2)
  {
    const uint8_t * in_top = in.data + y*in.cols;
    const uint8_t * in_bottom = in.data + (y+1)*in.cols;
    uint8_t * out_data = out.data + (y >> 1)*out.cols;
    for( int x = in.cols; x > 0 ; x-=16, in_top += 16, in_bottom += 16, out_data += 8)
    {
      uint8x8x2_t top  = vld2_u8( (const uint8_t *)in_top );
      uint8x8x2_t bottom = vld2_u8( (const uint8_t *)in_bottom );
      uint16x8_t sum = vaddl_u8( top.val[0], top.val[1] );
      sum = vaddw_u8( sum, bottom.val[0] );
      sum = vaddw_u8( sum, bottom.val[1] );
      uint8x8_t final_sum = vshrn_n_u16(sum, 2);
      vst1_u8(out_data, final_sum);
    }
  }
}
#endif

void
halfSample(const cv::Mat& in, cv::Mat& out)
{
  assert( in.rows/2==out.rows && in.cols/2==out.cols);
  assert( in.type()==CV_8U && out.type()==CV_8U);

  /*
#ifdef __SSE2__
  if(aligned_mem::is_aligned16(in.data) && aligned_mem::is_aligned16(out.data) && ((in.cols % 16) == 0))
  {
    halfSampleSSE2(in.data, out.data, in.cols, in.rows);
    return;
  }
#endif
*/
#ifdef __ARM_NEON__
  if( (in.cols % 16) == 0 )
  {
    halfSampleNEON(in, out);
    return;
  }
#endif

  const int stride = in.step.p[0];
  uint8_t* top = (uint8_t*) in.data;
  uint8_t* bottom = top + stride;
  uint8_t* end = top + stride*in.rows;
  const int out_width = out.cols;
  uint8_t* p = (uint8_t*) out.data;
  while (bottom < end)
  {
    for (int j=0; j<out_width; j++)
    {
      *p = static_cast<uint8_t>( (uint16_t (top[0]) + top[1] + bottom[0] + bottom[1])/4 );
      p++;
      top += 2;
      bottom += 2;
    }
    top += stride;
    bottom += stride;
  }
}

void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr)
{
  pyr.resize(n_levels);
  pyr[0] = img_level_0;
  for(int i=1; i<n_levels; ++i)
  {
    pyr[i] = cv::Mat(pyr[i-1].rows/2, pyr[i-1].cols/2, CV_8U);
    halfSample(pyr[i-1], pyr[i]);
  }
}

template<class T>
T getMedian(vector<T>& data_vec)
{
  assert(!data_vec.empty());
  typename vector<T>::iterator it = data_vec.begin()+floor(data_vec.size()/2);
  nth_element(data_vec.begin(), it, data_vec.end());
  return *it;
}

bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min)
{
  vector<double> depth_vec;
  depth_vec.reserve(frame.fts_.size());
  depth_min = std::numeric_limits<double>::max();
  for(auto it=frame.fts_.begin(), ite=frame.fts_.end(); it!=ite; ++it)
  {
    if((*it)->point != NULL)
    {
      const double z = frame.w2f((*it)->point->pos_).z();
      depth_vec.push_back(z);
      depth_min = fmin(z, depth_min);
    }
  }
  if(depth_vec.empty())
  {
    SVO_WARN_STREAM("Cannot set scene depth. Frame has no point-observations!");
    return false;
  }
  depth_mean = getMedian(depth_vec);
  return true;
}

} // namespace frame_utils
} // namespace svo
