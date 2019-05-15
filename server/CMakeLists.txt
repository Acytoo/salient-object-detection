add_library(
  basic STATIC  # SHARED
  src/basic/block.h
  src/basic/graph.h
  src/basic/disjoint_set.h
  src/basic/image_operations.h
  src/basic/image_operations.cc
  src/basic/segment_graph.h
  src/basic/segment_image.h
  src/basic/segment_image.cc
  src/basic/ytplatform.h
  src/basic/file_linux.h
  src/basic/file_win.h
  src/basic/basic_functions_demo.h
  src/basic/basic_functions_demo.cc
  src/basic/gaussian_blur.h
  )

add_library(
  saliency STATIC  # SHARED
  src/saliency/saliency_region_contrast.h
  src/saliency/saliency_region_contrast.cc
  src/saliency/some_definition.h
  src/saliency/some_definition.cc
  )

target_include_directories(basic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(saliency PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
