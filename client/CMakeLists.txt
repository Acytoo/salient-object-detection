find_package(Qt5Widgets)
find_package(Qt5Core)
find_package(Qt5Gui)


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(
  simple_gui STATIC
  simple_gui/main.cc
  simple_gui/mainwindow.h
  simple_gui/mainwindow.cc
  simple_gui/mainwindow.ui
  )

target_link_libraries (simple_gui Qt5::Widgets saliency basic)

target_include_directories(simple_gui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/simple_gui")


set(TARGET_EXECUTABLE test_no_gui.out)

add_executable(${TARGET_EXECUTABLE} run_no_gui.cc)

target_link_libraries(${TARGET_EXECUTABLE} PRIVATE ${OpenCV_LIBS} saliency basic)
