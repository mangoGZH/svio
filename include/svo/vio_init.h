//
// Created by gzh on 20-4-16.
//

#ifndef SVO_EDGELETE_LIVE_VIO_INIT_H
#define SVO_EDGELETE_LIVE_VIO_INIT_H

#include <svo/global.h>
#include <svo/frame_handler_mono.h>
#include <svo/frame.h>
#include <svo/map.h>
#include <svo/feature.h>
#include <svo/point.h>

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
//#include <numeric>

using namespace std;
using namespace cv;

namespace  svo{

    class Point;
    class Feature;
    class Map;
    class Frame;
    class FrameHandlerMono;

    class KeyFrameInit{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW  //内存对齐

        KeyFrameInit *prev_KeyFrame;
        IMUPreintegrator IMUPreInt;
        vector<IMUData> kfIMUData;
        //Mat Twc;
        Vector3d bg;
        double TimeStamp;

        KeyFrameInit(Frame& kf):
            prev_KeyFrame( NULL), IMUPreInt( kf.kfimuPreint),
            kfIMUData( kf.Get_kfIMUData()), bg(0,0,0) ,TimeStamp( kf.timestamp_){}

        //ComputePreInt between last KF and current KF
        void ComputePreInt();
    };



    class VioInitialization{
    public:
        VioInitialization (svo::FrameHandlerMono* vo);
        vector<double> listscale;

        void run();

        void Record_file();
        //Vector3d OptimizeInitialGyroBias(const vector<cv::Mat>& vTwc, const vector<IMUPreintegrator>& vImuPreInt);
        Vector3d solveGyroscopeBias(const vector<cv::Mat> &Pose_c_w, const vector<IMUPreintegrator> &ImuPreInt,
                                    const Matrix4d T_bc);

    private:

        int gzh;
        int lx;

        bool TryVioInit();
        svo::FrameHandlerMono* _vo;
        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

        double thresh_1 = 1e-03; //1e-04//5e-05
        double thresh_2 = 1e-03;
        int count;

        int last_kf_id;
        double s_convergence_time;    //gzh add 2020-1-14 to recored the Scale convergence time
        bool FirstTry;
        double StartTime;
        double VINSInitScale;

        cv::Mat GravityVec; // gravity vector in world frame
        cv::Mat Rwi_Init;

    };

}
#endif //SVO_EDGELETE_LIVE_VIO_INIT_H
