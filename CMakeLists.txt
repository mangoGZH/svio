################################################################################
# user build settings

SET(TRACE TRUE)
#SET(HAVE_G2O false)   #TRUE
SET(HAVE_G2O true)   #TRUE

SET(DEBUG_OUTPUT TRUE) # Only relevant if build without ROS

################################################################################
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3) # 指定 cmake 最低编译版本

SET(PROJECT_NAME svo_edgelete_live)
PROJECT(${PROJECT_NAME})        # 指定工程名

SET(CMAKE_BUILD_TYPE Release)   # 设定编译模式
#SET(CMAKE_BUILD_TYPE debug)    # Release, RelWithDebInfo , Debug

SET(CMAKE_VERBOSE_MAKEFILE OFF)    # 想看到更多cmake编译时候的信息，置为ON，否则为OFF.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")   # 定义自己的.cmake模块所在的路径,可用INCLUDE命令来调用的模块

# Set definitions
IF(TRACE)
  ADD_DEFINITIONS(-DSVO_TRACE)
ENDIF()

IF(DEBUG_OUTPUT)
  ADD_DEFINITIONS(-DSVO_DEBUG_OUTPUT)
ENDIF()

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
#SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
#CMAKE_CXX_FLAGS 设置C++编译选项
IF(DEFINED ENV{ARM_ARCHITECTURE})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages 
FIND_PACKAGE(OpenCV 2.4 REQUIRED)
FIND_PACKAGE(Eigen REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)

FIND_PACKAGE(Pangolin REQUIRED)

#FIND_PACKAGE(fast REQUIRED)
#FIND_PACKAGE(vikit_common REQUIRED)

#使用SET命令重新定义EXECUTABLE_OUTPUT_PATH和LIBRARY_OUTPUT_PATH变量来指定最终的二进制文件的位置
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)  # 重新定义目标二进制可执行文件的存放位置
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)     # 目标链接库文件 .so 动态库的存放位置


# Include dirs
INCLUDE_DIRECTORIES(   #向工程添加 头文件路径及库路径
  include
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

# Set link libraries
LIST(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Set sourcefiles
LIST(APPEND SOURCEFILES
  src/frame_handler_mono.cpp
  src/frame_handler_base.cpp
  src/frame.cpp
  src/point.cpp
  src/map.cpp
  src/pose_optimizer.cpp
  src/initialization.cpp
  src/matcher.cpp
  src/reprojector.cpp
  src/feature_alignment.cpp
  src/feature_detection.cpp
  src/depth_filter.cpp
  src/config.cpp

  src/camera_model.cpp
  src/sparse_align.cpp
  src/debug.cpp
  src/math_utils.cpp
  src/homography.cpp
  src/robust_cost.cpp
  src/fast_10_score.cpp
  src/fast_nonmax_3x3.cpp
  src/fast_10.cpp
  src/faster_corner_10_sse.cpp

  src/slamviewer.cpp
  src/five-point.cpp
  src/precomp.cpp
  src/modelest.cpp
  )

IF(HAVE_G2O)
  ADD_DEFINITIONS(-DUSE_BUNDLE_ADJUSTMENT)
ENDIF()
# Add g2o if available
IF(HAVE_G2O)
  FIND_PACKAGE(G2O REQUIRED)

  message("g2o is found: ${G2O_INCLUDE_DIR}")

  INCLUDE_DIRECTORIES(
    ${G2O_INCLUDE_DIR} #/usr/local/include/g2o
    /usr/include/suitesparse # for cholmod
  )

  LIST(APPEND LINK_LIBS
  ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE}
   ${G2O_SOLVER_PCG} ${G2O_TYPES_SBA} cholmod cxsparse
 )
  #LIST(APPEND LINK_LIBS  g2o_core_d  g2o_solver_csparse_d  g2o_csparse_extension_d g2o_types_sba_d g2o_solver_dense_d g2o_stuff_d g2o_parser_d g2o_solver_pcg_d  cholmod cxsparse )
  LIST(APPEND SOURCEFILES src/bundle_adjustment.cpp)
ENDIF(HAVE_G2O)

# Create svo library
ADD_LIBRARY(svo SHARED ${SOURCEFILES} include/svo/Data_EuRoc.h src/Data_EuRoc.cpp include/svo/Converter.h src/Converter.cpp include/svo/vio_init.h src/vio_init.cpp src/IMU/imudata.cpp src/IMU/IMUPreintegrator.cpp src/IMU/NavState.cpp)
TARGET_LINK_LIBRARIES(svo ${LINK_LIBS}   ${Pangolin_LIBRARIES})

################################################################################
# TESTS
#生成可执行文件
#ADD_EXEUTABLE(test_pipeline test/test_pipeline.cpp)
#TARGET_LINK_LIBRARIES(test_pipeline svo)

ADD_EXECUTABLE(test_pipel_euroc test/test_pipel_euroc.cpp)
TARGET_LINK_LIBRARIES(test_pipel_euroc svo)

#ADD_EXECUTABLE(test_live_vo test/test_live_vo.cpp)
#TARGET_LINK_LIBRARIES(test_live_vo svo)
