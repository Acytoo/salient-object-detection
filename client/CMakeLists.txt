find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5Core)
find_package(Qt5Gui)


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(
  salient_object_detectionW STATIC
  salient_object_detectionW/main.cc
  salient_object_detectionW/startwindow.h
  salient_object_detectionW/startwindow.cc
  salient_object_detectionW/startwindow.ui
  salient_object_detectionW/process_single_page.h
  salient_object_detectionW/process_single_page.cc
  salient_object_detectionW/process_single_page.ui
  salient_object_detectionW/process_multiple_page.h
  salient_object_detectionW/process_multiple_page.cc
  salient_object_detectionW/process_multiple_page.ui
  salient_object_detectionW/basic_image_operations.h
  salient_object_detectionW/basic_image_operations.cc
  salient_object_detectionW/basic_image_operations.ui
  )

target_link_libraries (salient_object_detectionW Qt5::Widgets basic saliency)

target_include_directories(salient_object_detectionW PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/salient_object_detectionW")




set(TARGET_EXECUTABLE salient_object_detection.out)

add_executable(${TARGET_EXECUTABLE} salient_object_detection.cc)

target_link_libraries(${TARGET_EXECUTABLE} PRIVATE ${OpenCV_LIBS} saliency basic)

set(TARGET_EXECUTABLE test_segment.out)

add_executable(${TARGET_EXECUTABLE} test_segment.cc)

target_link_libraries(${TARGET_EXECUTABLE} PRIVATE ${OpenCV_LIBS} saliency basic)
