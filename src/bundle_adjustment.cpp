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

//#include <vikit/math_utils.h>
//#include <boost/thread.hpp>
//#include <g2o/core/sparse_optimizer.h>
//#include <g2o/core/block_solver.h>
//#include <g2o/core/solver.h>
//#include <g2o/core/robust_kernel_impl.h>
//#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/types/sba/types_six_dof_expmap.h>
//#include <g2o/solvers/structure_only/structure_only_solver.h>
//#include <svo/bundle_adjustment.h>
//#include <svo/frame.h>
//#include <svo/feature.h>
//#include <svo/point.h>
//#include <svo/config.h>
//#include <svo/map.h>
#include <boost/thread.hpp>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>

#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <svo/bundle_adjustment.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/config.h>
#include <svo/map.h>
#include <svo/math_lib.h>
#include <svo/frame_handler_base.h>
#include <svo/frame_handler_mono.h>
#include <svo/depth_filter.h>
#include "../src/IMU/g2otypes.h"

#define SCHUR_TRICK 1

namespace svo {
namespace ba {

void twoViewBA(
    Frame* frame1,
    Frame* frame2,
    double reproj_thresh,
    Map* map)
{
  // scale reprojection threshold in pixels to unit plane
  reproj_thresh /= frame1->cam_->errorMultiplier2();

  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  size_t v_id = 0;

  // New Keyframe Vertex 1: This Keyframe is set to fixed!
  g2oFrameSE3* v_frame1 = createG2oFrameSE3(frame1, v_id++, true);
  optimizer.addVertex(v_frame1);

  // New Keyframe Vertex 2
  g2oFrameSE3* v_frame2 = createG2oFrameSE3(frame2, v_id++, false);
  optimizer.addVertex(v_frame2);

  // Create Point Vertices
  for(Features::iterator it_ftr=frame1->fts_.begin(); it_ftr!=frame1->fts_.end(); ++it_ftr)
  {
    Point* pt = (*it_ftr)->point;
    if(pt == NULL)
      continue;
    g2oPoint* v_pt = createG2oPoint(pt->pos_, v_id++, false);
    optimizer.addVertex(v_pt);
    pt->v_pt_ = v_pt;
    g2oEdgeSE3* e = createG2oEdgeSE3(v_frame1, v_pt, project2d((*it_ftr)->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame1, *it_ftr)); // TODO feature now links to frame, so we can simplify edge container!

    // find at which index the second frame observes the point
    Feature* ftr_frame2 = pt->findFrameRef(frame2);
    e = createG2oEdgeSE3(v_frame2, v_pt, project2d(ftr_frame2->f), true, reproj_thresh*Config::lobaRobustHuberWidth());
    optimizer.addEdge(e);
    edges.push_back(EdgeContainerSE3(e, frame2, ftr_frame2));
  }

  // Optimization
  double init_error, final_error;
  runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);
  printf("2-View BA: Error before/after = %f / %f\n", init_error, final_error);

  // Update Keyframe Positions
  frame1->T_f_w_.rotation_matrix() = v_frame1->estimate().rotation().toRotationMatrix();
  frame1->T_f_w_.translation() = v_frame1->estimate().translation();
  frame2->T_f_w_.rotation_matrix() = v_frame2->estimate().rotation().toRotationMatrix();
  frame2->T_f_w_.translation() = v_frame2->estimate().translation();

  // Update Mappoint Positions
  for(Features::iterator it=frame1->fts_.begin(); it!=frame1->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
     continue;
    (*it)->point->pos_ = (*it)->point->v_pt_->estimate();
    (*it)->point->v_pt_ = NULL;
  }

  // Find Mappoints with too large reprojection error
  const double reproj_thresh_squared = reproj_thresh*reproj_thresh;
  size_t n_incorrect_edges = 0;
  for(list<EdgeContainerSE3>::iterator it_e = edges.begin(); it_e != edges.end(); ++it_e)
    if(it_e->edge->chi2() > reproj_thresh_squared)
    {
      if(it_e->feature->point != NULL)
      {
        map->safeDeletePoint(it_e->feature->point);
        it_e->feature->point = NULL;
      }
      ++n_incorrect_edges;
    }

  printf("2-View BA: Wrong edges =  %zu\n", n_incorrect_edges);
}

void localBA(
    Frame* center_kf,          //Current frame.
    set<FramePtr>* core_kfs,
    Map* map,
    size_t& n_incorrect_edges_1,
    size_t& n_incorrect_edges_2,
    double& init_error,
    double& final_error)
{
  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  set<Point*> mps;
  list<Frame*> neib_kfs;
  size_t v_id = 0;
  size_t n_mps = 0;
  size_t n_fix_kfs = 0;
  size_t n_var_kfs = 1;
  size_t n_edges = 0;
  n_incorrect_edges_1 = 0;
  n_incorrect_edges_2 = 0;

  // Add all core keyframes
  for(set<FramePtr>::iterator it_kf = core_kfs->begin(); it_kf != core_kfs->end(); ++it_kf)
  {
    g2oFrameSE3* v_kf = createG2oFrameSE3(it_kf->get(), v_id++, false);
    (*it_kf)->v_kf_ = v_kf;
    ++n_var_kfs;
    assert(optimizer.addVertex(v_kf));

    // all points that the core keyframes observe are also optimized:
    for(Features::iterator it_pt=(*it_kf)->fts_.begin(); it_pt!=(*it_kf)->fts_.end(); ++it_pt)
      if((*it_pt)->point != nullptr)
        mps.insert((*it_pt)->point);
  }
    cout<<"localba mps.size() is:\t"<< mps.size() <<endl;

  // Now go throug all the points and add a measurement. Add a fixed neighbour
  // Add a fixed neighbour Keyframe if it is not in the set of core kfs
  double reproj_thresh_2 = Config::lobaThresh() / center_kf->cam_->errorMultiplier2();
  double reproj_thresh_1 = Config::poseOptimThresh() / center_kf->cam_->errorMultiplier2();
  double reproj_thresh_1_squared = reproj_thresh_1*reproj_thresh_1;
  for(set<Point*>::iterator it_pt = mps.begin(); it_pt!=mps.end(); ++it_pt)
  {
    // Create point vertex
    g2oPoint* v_pt = createG2oPoint((*it_pt)->pos_, v_id++, false);
    (*it_pt)->v_pt_ = v_pt;
    assert(optimizer.addVertex(v_pt));
    ++n_mps;

    // Add edges
    list<Feature*>::iterator it_obs=(*it_pt)->obs_.begin();
    while(it_obs!=(*it_pt)->obs_.end())
    {
      Vector2d error = project2d((*it_obs)->f) - project2d((*it_obs)->frame->w2f((*it_pt)->pos_));

      if((*it_obs)->frame->v_kf_ == nullptr)
      {
        // frame does not have a vertex yet -> it belongs to the neib kfs and
        // is fixed. create one:
        g2oFrameSE3* v_kf = createG2oFrameSE3((*it_obs)->frame, v_id++, true);
        (*it_obs)->frame->v_kf_ = v_kf;
        ++n_fix_kfs;
        assert(optimizer.addVertex(v_kf));
        neib_kfs.push_back((*it_obs)->frame);
      }

      // create edge
      g2oEdgeSE3* e = createG2oEdgeSE3((*it_obs)->frame->v_kf_, v_pt,
                                       project2d((*it_obs)->f),
                                       true,
                                       reproj_thresh_2*Config::lobaRobustHuberWidth(),
                                       1.0 / (1<<(*it_obs)->level));
      assert(optimizer.addEdge(e));
      edges.push_back(EdgeContainerSE3(e, (*it_obs)->frame, *it_obs));
      ++n_edges;
      ++it_obs;
    }
  }
    cout<<"localba neib_kfs.size() is\t"<< neib_kfs.size()<<endl;
  // structure only
  g2o::StructureOnlySolver<3> structure_only_ba;
  g2o::OptimizableGraph::VertexContainer points;
  for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
  {
    g2o::OptimizableGraph::Vertex* v = dynamic_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() == 3 && v->edges().size() >= 2)
        points.push_back(v);
  }
  structure_only_ba.calc(points, 10);

  // Optimization
  if(Config::lobaNumIter() > 0)
    runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);

  // Update Keyframes
  for(set<FramePtr>::iterator it = core_kfs->begin(); it != core_kfs->end(); ++it)
  {
    (*it)->T_f_w_ = SE3( (*it)->v_kf_->estimate().rotation(),
                         (*it)->v_kf_->estimate().translation());
    (*it)->v_kf_ = nullptr;
  }

  for(list<Frame*>::iterator it = neib_kfs.begin(); it != neib_kfs.end(); ++it)
    (*it)->v_kf_ = nullptr;

  // Update Mappoints
  for(set<Point*>::iterator it = mps.begin(); it != mps.end(); ++it)
  {
    (*it)->pos_ = (*it)->v_pt_->estimate();
    (*it)->v_pt_ = nullptr;
  }

  // Remove Measurements with too large reprojection error
  double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
  for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
  {
    if(it->edge->chi2() > reproj_thresh_2_squared) //*(1<<it->feature_->level))
    {
      map->removePtFrameRef(it->frame, it->feature);
      ++n_incorrect_edges_2;
    }
  }

  // TODO: delete points and edges!
  init_error = sqrt(init_error)*center_kf->cam_->errorMultiplier2();
  final_error = sqrt(final_error)*center_kf->cam_->errorMultiplier2();
}

//void windowBA(
//    Frame *cur_kf,
//    vector <FramePtr> *window_kfs,
//    Map *map,
//    Vector3d gw,
//    Matrix4d Tbc,
//    size_t &n_incorrect_edges_1,
//    size_t &n_incorrect_edges_2,
//    double &init_error,
//    double &final_error)

    void windowBA(
            Frame *cur_kf,
            list <FramePtr> *win_kfs,
            Map *map,
            Vector3d gw,
            Matrix4d Tbc,
            size_t &n_incorrect_edges_1,
            size_t &n_incorrect_edges_2,
            double &init_error,
            double &final_error){

    // Setup optimizer
    g2o::SparseOptimizer optimizer;

    g2o::BlockSolverX ::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(
            std::unique_ptr<g2o::BlockSolverX::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
            std::unique_ptr<g2o::BlockSolverX>(solver_ptr));

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::CameraParameters *cam_params = new g2o::CameraParameters(1.0, Vector2d(0., 0.), 0.);
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params)) { assert(false); }

    // set initial
    int maxKFid = 0;
    list<EdgeContainerSE3> vp_edge;
    list<pair<Frame *, Feature *> > bad_edges;
    set<Point*> mps;
    list<Frame*> FixedKfs;
    size_t v_id = 0;
    size_t n_mps = 0;
    n_incorrect_edges_1 = 0;
    n_incorrect_edges_2 = 0;
    double reproj_thresh_2 = Config::lobaThresh() / cur_kf->cam_->errorMultiplier2();  //loba_thresh(2.0),
    Vector3d GravityVec = gw;
    cout<<"window BA computing...."<<endl;

    /// Add all window keyframes  ---> Set window KeyFrame vertices
    int cnt = 1;
    for(list<FramePtr>::iterator it_kf = win_kfs->begin(); it_kf != win_kfs->end(); ++it_kf, ++cnt){
        Frame *pKF = it_kf->get();
        int id_KF = (cnt-1)*2;
        //Vertex of PVR
        {
            auto *vPVR = new g2o::VertexNavStatePVR();
            vPVR->setEstimate(pKF->GetNavState());
            vPVR->setId(id_KF);
            vPVR->setFixed(false);
            pKF->v_PVR_ = vPVR;
            optimizer.addVertex(vPVR);
//            cout<<"add KF : "<<pKF->kf_id_ <<" v_PVR_ to vertex  ,the vertex id is:"<<id_KF<<endl;
        }
        // Vertex of Bias
        {
            auto * vBias = new g2o::VertexNavStateBias();
            vBias->setEstimate(pKF->GetNavState());
            vBias->setId(id_KF+1);
            vBias->setFixed(false);
            pKF->v_Bias_ = vBias;
            assert(optimizer.addVertex(vBias));
//            cout<<"add KF : "<<pKF->kf_id_ <<" v_Bias_ to vertex  ,the vertex id is:"<<id_KF+1<<endl;
        }
        maxKFid = max( (id_KF+1), maxKFid);
        //cout<<"maxKFid is: "<< maxKFid<<endl;
    }

    ///get map point list observed by window_kfs
    for(list<FramePtr>::iterator it_kf = win_kfs->begin(); it_kf!= win_kfs->end(); ++it_kf){
        for(Features::iterator it_ftr = (*it_kf)->fts_.begin(); it_ftr != (*it_kf)->fts_.end(); ++it_ftr)
            if((*it_ftr)->point != nullptr)
                mps.insert((*it_ftr)->point);
    }
    cout<<"mps.size() is :"<<mps.size()<<endl;

    Frame* pKFPrevwindow = win_kfs->front()->PrevKeyFrame;// Add the KeyFrame before local window.
    if(pKFPrevwindow) {
        if (pKFPrevwindow != nullptr) {
            FixedKfs.push_back(pKFPrevwindow);
            //cout<<"add pKFPrevLocal:"<<pKFPrevwindow->kf_id_<<"to  fixedKF,and the fixedkf size is:"<< FixedKfs.size()<<endl;
            pKFPrevwindow->putKFtofixedKF();
        }
    }else {cerr<<"pKFPrevLocal is NULL?"<<endl;}
    cout<<"fixed kf vertics size():"<<FixedKfs.size()<<endl;

    /// Covisible KeyFrames (set fixedKF)
    int count0 = 0;
    int count = 0;
    for(set<Point*>::iterator it_pt = mps.begin(); it_pt != mps.end(); ++it_pt) {
        if ((*it_pt) == NULL) continue;
        list<Feature *> it_obs = (*it_pt)->obs_;

//        ///select fixedkf method 1: put all Covisible KeyFrames into fixedkf list
//        for (list<Feature *>::iterator fit = it_obs.begin(), fend = it_obs.end(); fit != fend; fit++) {
//            count0++;
//            if (((*fit)->frame->v_PVR_ == nullptr) &&
//                !(*fit)->frame->is_infixed_kf) {  //if the kf not in_window and not in fixedkf
//
//                FixedKfs.push_back((*fit)->frame);
//                (*fit)->frame->putKFtofixedKF();
//                //cout << "FixedKfs id ia:\t" << (*fit)->frame->kf_id_ << endl;
//            }
//        }
//        cout << "count0: \t" << count0 << endl;

        ///select fixedkf method 2: KeyFrames which can observe points more than FixedKFMinFtrNum()
        for (list<Feature *>::iterator fit = it_obs.begin(), fend = it_obs.end(); fit != fend; fit++) {
            (*fit)->frame->winmap_obs_ftr++;
            if (((*fit)->frame->v_PVR_ == nullptr)
                && ( !(*fit)->frame->is_infixed_kf )
                && ((*fit)->frame->winmap_obs_ftr > Config::FixedKFMinFtrNum())) //10
            {
                count++;
                FixedKfs.push_back((*fit)->frame);
                (*fit)->frame->putKFtofixedKF();
            }
        }
//        cout << "fixedkf count1:\t" << count << endl;

        ///select fixedkf method 3: put localBA KeyFrames(has flag) into fixedkf list
    }

    //把先前kf的 winmap_obs_ftr清零
    for(list<FramePtr>::iterator it_kf = map->keyframes_.begin(); it_kf != map->keyframes_.end(); ++it_kf) {
        Frame *pKF = it_kf->get();
        pKF->winmap_obs_ftr = 0;
    }

    ///set fixed kf vertics : including the PKFPreveLocal
    v_id = (maxKFid+1)/2;
    int id_kf = v_id*2;
    for(list<Frame*>::iterator it_kf = FixedKfs.begin(); it_kf != FixedKfs.end(); ++it_kf,++id_kf) {
        Frame *pKF = *it_kf;
        if (pKF == pKFPrevwindow) {
            // For Local-Window-Previous KeyFrame, add Bias vertex and PVR vertex
            auto *vNSPVR = new g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(pKF->GetNavState());
            vNSPVR->setId(id_kf);
            vNSPVR->setFixed(true);
            pKF->v_PVR_ = vNSPVR;
            optimizer.addVertex(vNSPVR);
//            cout << "add fixed pKFPrev : " << pKF->kf_id_ << " v_PVR_ to vertex  ,the vertex id is:" << id_kf << endl;
            auto *vNSBias = new g2o::VertexNavStateBias();
            vNSBias->setEstimate(pKF->GetNavState());
            vNSBias->setId(++id_kf);
            vNSBias->setFixed(true);
            pKF->v_Bias_ = vNSBias;
            assert(optimizer.addVertex(vNSBias));
//            cout << "add fixed pKFPrev : " << pKF->kf_id_ << " v_Bias_ to vertex  ,the vertex id is:" << id_kf << endl;
        } else {
            // For common fixed KeyFrames, only add PVR vertex
            auto *vNSPVR = new g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(pKF->GetNavState());
            vNSPVR->setId(id_kf);
            vNSPVR->setFixed(true);
            pKF->v_PVR_ = vNSPVR;
            optimizer.addVertex(vNSPVR);
//            cout << "add fixed KF : " << pKF->kf_id_ << " v_PVR_ to vertex  ,the vertex id is:" << id_kf << endl;
        }
        maxKFid = max(id_kf, maxKFid);
        //cout << "maxKFid is: " << maxKFid << endl;
    }
    //*************************windowba test debug********************2020-1-28
    static bool fopened = false;
    static ofstream FixedKFnum;
    string  filepath = "/home/gzh/SVIO_rebuild/svio_outfile/";
    if(!fopened) {
        FixedKFnum.open(filepath + "FixedKF.txt");
        if (FixedKFnum.is_open())
            fopened = true;
        else {
            cerr << "file open error in TryInitVIO" << endl;
            fopened = false;
        }
        FixedKFnum << std::fixed << std::setprecision(6);
    }
    FixedKFnum << cur_kf->timestamp_<<" "
               << cur_kf->kf_id_<<" "
               << FixedKfs.size()<<" "<<endl;

    cout<<"fixed kf vertics size():"<<FixedKfs.size()<<endl;
    //*******************************************************************

    /// Edges between KeyFrames in Window and
    /// Edges between 1st KeyFrame of  Window and its previous (fixed)KeyFrame - pKFPrevLocal
    vector<g2o::EdgeNavStatePVR*> v_EdgesNavStatePVR;
    vector<g2o::EdgeNavStateBias*> v_EdgesNavStateBias;
    const float thHuberNavStatePVR = sqrt(100*21.666);
    const float thHuberNavStateBias = sqrt(100*16.812);
    // Inverse covariance of bias random walk
    Matrix<double, 6, 6> InvCovBgaRW = Matrix<double, 6, 6>::Identity();
    InvCovBgaRW.topLeftCorner(3, 3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3, 3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE

    for(list<FramePtr>::iterator it_kf = win_kfs->begin(); it_kf != win_kfs->end(); ++it_kf) {
        Frame *pKF = it_kf->get();  // Current KF, store the IMU pre-integration between previous-current
        Frame *pKF0 = pKF->GetPrevKeyFrame(); // Previous KF

        // PVR edge
        auto *ePVR = new g2o::EdgeNavStatePVR();
        ePVR->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pKF0->v_PVR_));
        ePVR->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pKF->v_PVR_));
        ePVR->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pKF0->v_Bias_));
        ePVR->setMeasurement(pKF->kfimuPreint);

        Matrix9d InvConPVR = pKF->kfimuPreint.getCovPVPhi().inverse();
        ePVR->setInformation(InvConPVR);
        ePVR->SetParams(GravityVec);

        auto *rkPVR = new g2o::RobustKernelHuber;
        ePVR->setRobustKernel(rkPVR);
        rkPVR->setDelta(thHuberNavStatePVR);

        optimizer.addEdge(ePVR);
        v_EdgesNavStatePVR.push_back(ePVR);

        // Bias edge
        auto *eBias = new g2o::EdgeNavStateBias();
        eBias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex * >(pKF0->v_Bias_));
        eBias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex * >(pKF->v_Bias_));
        eBias->setMeasurement(pKF->kfimuPreint);
        eBias->setInformation(InvCovBgaRW / pKF->kfimuPreint.getDeltaTime());

        auto *rkBias = new g2o::RobustKernelHuber;
        eBias->setRobustKernel(rkBias);
        rkBias->setDelta(thHuberNavStateBias);

        optimizer.addEdge(eBias);
        v_EdgesNavStateBias.push_back(eBias);
    }
    /// Set MapPoint vertices
    const int nExpectedSize = (10 + FixedKfs.size()) * mps.size();
    vector<g2o::EdgeNavStatePVRPointXYZ *> vpEdges;
    vpEdges.reserve(nExpectedSize);

    vector<Frame *> vpEdgeKF;
    vpEdgeKF.reserve(nExpectedSize);

    vector<Point *> vpEdgeMapPoint;
    vpEdgeMapPoint.reserve(nExpectedSize);

    vector<Feature *>vpEdgeFeature;
    vpEdgeFeature.reserve(nExpectedSize);
    const float thHuberMono = sqrt(5.991);

    // Extrinsics
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);

    for(auto it_pt = mps.begin(); it_pt != mps.end(); ++it_pt) {

        Point* pMp = *it_pt;
        int mpVertexId = n_mps + maxKFid + 1;

       auto *v_pt = new g2o::g2oPoint();  // g2oPoint == VertexSBAPointXYZ
        v_pt->setEstimate((pMp->pos_));
        v_pt->setId(mpVertexId);
        v_pt->setFixed(true);         //false true
        v_pt->setMarginalized(true);
        (*it_pt)->v_pt_ = v_pt;
        assert(optimizer.addVertex(v_pt));

        ++n_mps;

        /// Add edges between KeyFrame and this MapPoint
        //obs_: References to keyframes which observe the point.
        for( list<Feature *>::iterator it_obs = pMp->obs_.begin(); it_obs != pMp->obs_.end(); it_obs++)
        {
            Frame *pKF = (*it_obs)->frame;
            if(pKF->v_PVR_ != NULL){
                auto *e = new g2o::EdgeNavStatePVRPointXYZ();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(v_pt));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pKF->v_PVR_));
                e->setMeasurement(project2d((*it_obs)->f));
                e->setInformation(Eigen::Matrix2d::Identity()*Config::Edgeweight_PVRPoint()
                                  / pow(1.2,(*it_obs)->level));
                                 // 1.0 / (1 << (*it_obs)->level))    //0.008  //*exp(-(*it_obs)->level)

                auto *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);
                e->SetParams(pKF->cam_->fx(),pKF->cam_->fy(),pKF->cam_->cx(),pKF->cam_->cy(),Rbc,Pbc);
                assert(optimizer.addEdge(e));

                vpEdges.push_back(e);
                vpEdgeKF.push_back(pKF);
                vpEdgeMapPoint.push_back(pMp);
                vpEdgeFeature.push_back((*it_obs));
            }
        }
    }
    cout<<"add mappoint vertex num: " << mps.size()
        << "\t add vpEdges num: "<< vpEdges.size()<<endl;

    // structure only
    g2o::StructureOnlySolver<3> structure_only_ba;
    g2o::OptimizableGraph::VertexContainer points;
    for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin();
             it != optimizer.vertices().end(); ++it) {
        auto *v = dynamic_cast<g2o::OptimizableGraph::Vertex *>(it->second);
        if (v->dimension() == 3 && v->edges().size() >= 2)
            points.push_back(v);
    }
    structure_only_ba.calc(points, 10);

    runSparseBAOptimizer(&optimizer, 10, init_error, final_error);

    // Check inlier observations
    bool check_inlier = true;
    if(check_inlier) {
        for (size_t i = 0, iend = v_EdgesNavStatePVR.size(); i < iend; i++) {
            auto *e1 = v_EdgesNavStatePVR[i];
            //cout<<"e1->chi2()"<<e1->chi2()<<endl;
        }
        for (size_t i = 0, iend = v_EdgesNavStateBias.size(); i < iend; i++) {
            auto *e2 = v_EdgesNavStateBias[i];
            // cout<<"e2->chi2()"<<e2->chi2()<<endl;
        }
        for (size_t i = 0, iend = vpEdges.size(); i < iend; i++) {
            auto *e3 = vpEdges[i];
            Point *pMP = vpEdgeMapPoint[i];
            //cout<<"e3->chi2()"<<e3->chi2()<<endl;
            // 判断条件1:基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
            // 判断条件2:用于检查构成该边的MapPoint在该相机坐标系下的深度是否为正？
            if (e3->chi2() > 5.991 || !e3->isDepthPositive()) {
                e3->setLevel(1);     //gzh fixed 2019-9-24
                //e->setLevel(0);   //setLevel为0，即下次不再对该边进行优化
            }
            e3->setRobustKernel(0);//因为剔除了错误的边，所以下次优化不再使用核函数
        }
        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
        //check_inlier = false;
    }

    vector<pair<Frame*, Feature*> > vToErase;
    vToErase.reserve(vpEdges.size());

    double PosePointchi2 = 0;
    for(size_t i=0, iend = vpEdges.size(); i<iend; ++i){
        auto *e4 = vpEdges[i];
        Feature *pFts = vpEdgeFeature[i];

        if(e4->chi2() > 5.991 || !e4->isDepthPositive()){
            Frame* pKFi = vpEdgeKF[i];
            vToErase.push_back(make_pair(pKFi, pFts));
        }
        PosePointchi2 += e4->chi2();
    }
    cout<<"vToErase.size() :"<<vToErase.size() <<endl;

    if( !vToErase.empty() ){
        for(size_t i=0; i< vToErase.size(); ++i){
            Frame *pKFi = vToErase[i].first;
            Feature* pFts = vToErase[i].second;
            map->removePtFrameRef(pKFi,pFts);
        }
    }

    /// Update Keyframes and clear g2o reserve
    for(list<FramePtr>::iterator it = win_kfs->begin(); it != win_kfs->end(); ++it){
        Frame* pKF = it->get();
        //update imustate
        auto *vNSPVR = static_cast<g2o::VertexNavStatePVR*>( pKF->v_PVR_);
        auto *vNSBias = static_cast<g2o::VertexNavStateBias*>( pKF->v_Bias_);
        // In optimized navstate, bias not changed, delta_bias not zero, should be added to bias
        const NavState& optPVRns = vNSPVR->estimate();
        const NavState& optBiasns = vNSBias->estimate();
        NavState primaryns = pKF->GetNavState();
        // Update NavState
        pKF->SetNavStatePos(optPVRns.Get_P());              //cout<<"pKF->SetNavStatePos:"<<optPVRns.Get_P() <<endl;
        pKF->SetNavStateVel(optPVRns.Get_V());              //cout<<"pKF->SetNavStateVel:"<<optPVRns.Get_V() <<endl;
        pKF->SetNavStateRot(optPVRns.Get_R());              //cout<<"pKF->SetNavStateRot:"<<optPVRns.Get_R() <<endl;
        pKF->SetNavStateDeltaBg(optBiasns.Get_dBias_Gyr()); //cout<<"updated delta bias gyr :"<<optBiasns.Get_dBias_Gyr() <<endl;
        pKF->SetNavStateDeltaBa(optBiasns.Get_dBias_Acc()); //cout<<"updated delta bias acc :"<<optBiasns.Get_dBias_Acc() <<endl;

        //update visual state
        cv::Mat tbc = Converter::toCvMat(Tbc);
        pKF->UpdatePoseFromNS(tbc);
        //delete g2o reserve
        pKF->v_PVR_  = nullptr;
        pKF->v_Bias_ = nullptr;
    }
    cout<<"after windowba---cur_kf->imuState.Get_P() :\n"<<cur_kf->imuState.Get_P() <<endl;
    cout<<"after windowba---cur_kf->pos()\n"<<cur_kf->pos() <<endl;

    for(auto it_kf = map->keyframes_.begin(); it_kf!= map->keyframes_.end(); ++it_kf){
        Frame* pKF = it_kf->get();
        pKF->is_infixed_kf = false;
        pKF->v_PVR_ = nullptr;
        pKF->v_Bias_ = nullptr;
    }
    //update map point position
    for(auto it_pt = mps.begin(); it_pt !=mps.end(); ++it_pt){
        Point* pMp = *it_pt;
        if(pMp == nullptr)
            continue;
        (*it_pt)->pos_ = (*it_pt)->v_pt_->estimate();
        (*it_pt)->v_pt_ = nullptr;
    }


//
}



void globalBA(Map* map)
{
  // init g2o
  g2o::SparseOptimizer optimizer;
  setupG2o(&optimizer);

  list<EdgeContainerSE3> edges;
  list< pair<FramePtr,Feature*> > incorrect_edges;

  // Go through all Keyframes
  size_t v_id = 0;
  double reproj_thresh_2 = Config::lobaThresh() / map->lastKeyframe()->cam_->errorMultiplier2();
  double reproj_thresh_1_squared = Config::poseOptimThresh() / map->lastKeyframe()->cam_->errorMultiplier2();
  reproj_thresh_1_squared *= reproj_thresh_1_squared;
  for(list<FramePtr>::iterator it_kf = map->keyframes_.begin();
      it_kf != map->keyframes_.end(); ++it_kf)
  {
    // New Keyframe Vertex
    g2oFrameSE3* v_kf = createG2oFrameSE3(it_kf->get(), v_id++, false);
    (*it_kf)->v_kf_ = v_kf;
    optimizer.addVertex(v_kf);
    for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
    {
      // for each keyframe add edges to all observed mapoints
      Point* mp = (*it_ftr)->point;
      if(mp == NULL)
        continue;
      g2oPoint* v_mp = mp->v_pt_;
      if(v_mp == NULL)
      {
        // mappoint-vertex doesn't exist yet. create a new one:
        v_mp = createG2oPoint(mp->pos_, v_id++, false);
        mp->v_pt_ = v_mp;
        optimizer.addVertex(v_mp);
      }

      // Due to merging of mappoints it is possible that references in kfs suddenly
      // have a very large reprojection error which may result in distorted results.
      Vector2d error = project2d((*it_ftr)->f) - project2d((*it_kf)->w2f(mp->pos_));
      if(error.squaredNorm() > reproj_thresh_1_squared)
        incorrect_edges.push_back(pair<FramePtr,Feature*>(*it_kf, *it_ftr));
      else
      {
        g2oEdgeSE3* e = createG2oEdgeSE3(v_kf, v_mp, project2d((*it_ftr)->f),
                                         true,
                                         reproj_thresh_2*Config::lobaRobustHuberWidth());

        edges.push_back(EdgeContainerSE3(e, it_kf->get(), *it_ftr));
        optimizer.addEdge(e);
      }
    }
  }

  // Optimization
  double init_error=0.0, final_error=0.0;
  if(Config::lobaNumIter() > 0)
    runSparseBAOptimizer(&optimizer, Config::lobaNumIter(), init_error, final_error);

  // Update Keyframe and MapPoint Positions
  for(list<FramePtr>::iterator it_kf = map->keyframes_.begin();
        it_kf != map->keyframes_.end(); ++it_kf)
  {
    (*it_kf)->T_f_w_ = SE3( (*it_kf)->v_kf_->estimate().rotation(),
                            (*it_kf)->v_kf_->estimate().translation());
    (*it_kf)->v_kf_ = NULL;
    for(Features::iterator it_ftr=(*it_kf)->fts_.begin(); it_ftr!=(*it_kf)->fts_.end(); ++it_ftr)
    {
      Point* mp = (*it_ftr)->point;
      if(mp == NULL)
        continue;
      if(mp->v_pt_ == NULL)
        continue;       // mp was updated before
      mp->pos_ = mp->v_pt_->estimate();
      mp->v_pt_ = NULL;
    }
  }

  // Remove Measurements with too large reprojection error
  for(list< pair<FramePtr,Feature*> >::iterator it=incorrect_edges.begin();
      it!=incorrect_edges.end(); ++it)
    map->removePtFrameRef(it->first.get(), it->second);

  double reproj_thresh_2_squared = reproj_thresh_2*reproj_thresh_2;
  for(list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it)
  {
    if(it->edge->chi2() > reproj_thresh_2_squared)
    {
      map->removePtFrameRef(it->frame, it->feature);
    }
  }
}

void setupG2o(g2o::SparseOptimizer * optimizer)
{
  // TODO: What's happening with all this HEAP stuff? Memory Leak?
  optimizer->setVerbose(false);

#if SCHUR_TRICK
  // solver
  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
  linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();

//  g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
//  g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(
            std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
            std::unique_ptr<g2o::BlockSolver_6_3>(solver_ptr));
#else
  g2o::BlockSolverX::LinearSolverType * linearSolver;
  linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();
  //linearSolver = new g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
  g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
#endif

  solver->setMaxTrialsAfterFailure(5);
  optimizer->setAlgorithm(solver);

  // setup camera
  g2o::CameraParameters * cam_params = new g2o::CameraParameters(1.0, Vector2d(0.,0.), 0.);
  cam_params->setId(0);
  if (!optimizer->addParameter(cam_params)) {
    assert(false);
  }
}

void runSparseBAOptimizer(g2o::SparseOptimizer* optimizer,
                     unsigned int num_iter,
                     double& init_error, double& final_error)
{
  optimizer->initializeOptimization();
  optimizer->computeActiveErrors();
  init_error = optimizer->activeChi2();
  optimizer->optimize(num_iter);
  final_error = optimizer->activeChi2();
}

g2oFrameSE3*
createG2oFrameSE3(Frame* frame, size_t id, bool fixed)
{
  g2oFrameSE3* v = new g2oFrameSE3();
  v->setId(id);
  v->setFixed(fixed);
  v->setEstimate(g2o::SE3Quat(frame->T_f_w_.unit_quaternion(), frame->T_f_w_.translation()));
  return v;
}

g2oPoint*
createG2oPoint(Vector3d pos,
               size_t id,
               bool fixed)
{
  g2oPoint* v = new g2oPoint();
  v->setId(id);
#if SCHUR_TRICK
  v->setMarginalized(true);
#endif
  v->setFixed(fixed);
  v->setEstimate(pos);
  return v;
}

g2oEdgeSE3*
createG2oEdgeSE3( g2oFrameSE3* v_frame,
                  g2oPoint* v_point,
                  const Vector2d& f_up,
                  bool robust_kernel,
                  double huber_width,
                  double weight)
{
  g2oEdgeSE3* e = new g2oEdgeSE3();
  e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
  e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
  e->setMeasurement(f_up);
  e->information() = weight * Eigen::Matrix2d::Identity(2,2);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber();      // TODO: memory leak
  rk->setDelta(huber_width);
  e->setRobustKernel(rk);
  e->setParameterId(0, 0); //old: e->setId(v_point->id());
  return e;
}

} // namespace ba
} // namespace svo
