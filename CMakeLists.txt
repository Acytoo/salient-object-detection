
# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.13)
project(salient_object_detection VERSION 0.2.2) # changed to 'object'


if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

if(LINUX)
  # for Linux, BSD, Solaris, Minix
  message(STATUS ">>> Linux")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

else()
  message(STATUS ">>> Windows")
  # WINDOWS ONLY
  # set your opencv dir first
  set(OpenCV_DIR  "E:/Software/opencv/build/x64/vc14/lib")
  set(CMAKE_PREFIX_PATH C:/Qt/Qt5.12.1/5.12.1/msvc2017_64/lib/cmake)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_subdirectory(server)
add_subdirectory(client)

set(TARGET_EXECUTABLE_GUI salient_object_detectionW.out)
add_executable(${TARGET_EXECUTABLE_GUI} WIN32 client/salient_object_detectionW/main.cc) # remove 'WIN32' to show console
target_link_libraries(${TARGET_EXECUTABLE_GUI} PRIVATE ${OpenCV_LIBS} saliency basic cluster salient_object_detectionW)
