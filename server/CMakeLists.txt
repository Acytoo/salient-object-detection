add_library(
  basic STATIC
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
  )

add_library(
  saliency STATIC
  src/saliency/saliency_cut.h
  src/saliency/saliency_cut.cc
  src/saliency/saliency_region_contrast.h
  src/saliency/saliency_region_contrast.cc
  src/saliency/some_definition.h
  src/saliency/some_definition.cc
  )

add_library(
  cluster STATIC
  src/cluster/gaussian_mixture_models.h
  src/cluster/gaussian_mixture_models.cc
  )

target_include_directories(basic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(cluster PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_include_directories(saliency PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
