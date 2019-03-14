# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

cmake_minimum_required(VERSION 3.13)
project(salient_region_detection VERSION 0.1.3)


if(UNIX AND NOT APPLE)
  set(LINUX TRUE)
endif()

if(LINUX)
  # for Linux, BSD, Solaris, Minix
  message(STATUS ">>> Linux")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

else()
  message(STATUS ">>> Not Linux")
  # basically is WINDOWS
  # set your opencv dir first
  set(OpenCV_DIR  "E:/Software/opencv/build/x64/vc14/lib")
  set(CMAKE_PREFIX_PATH C:/Qt/Qt5.12.1/5.12.1/msvc2017_64/lib/cmake)
endif()


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(server)
add_subdirectory(client)

set(TARGET_EXECUTABLE_GUI simple_gui.run)
add_executable(${TARGET_EXECUTABLE_GUI} client/simple_gui/main.cc)
target_link_libraries(${TARGET_EXECUTABLE_GUI} PRIVATE ${OpenCV_LIBS} saliency basic cluster simple_gui)
