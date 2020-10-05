//
// Created by gzh on 20-4-14.
//

#ifndef SVO_EDGELETE_LIVE_CONVERTER_H
#define SVO_EDGELETE_LIVE_CONVERTER_H

#include<opencv2/core/core.hpp>
#include<Eigen/Dense>
#include"g2o/types/sba/types_six_dof_expmap.h"
#include"g2o/types/sim3/types_seven_dof_expmap.h"
//#include "../src/IMU/IMUPreintegrator.h"
//#include "../src/IMU/NavState.h"
#include "global.h"
namespace svo{

    class  Converter{
    public:
        //static void updateNS(NavState& ns, const IMUPreintegrator& imupreint, const Vector3d& gw);
        static cv::Mat toCvMatInverse(const cv::Mat &T12);

    public:
        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
        static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
        static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
        static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
        static cv::Mat toCvMat(const Eigen::Matrix3d &m);
        static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
        static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

        static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
        static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
        static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);
        static Eigen::Matrix<double,4,4> toMatrix4d(const cv::Mat &cvMat4);

        static std::vector<float> toQuaternion(const cv::Mat &M);
    };

}
#endif //SVO_EDGELETE_LIVE_CONVERTER_H
