cmake_minimum_required(VERSION 3.8)
project(proxemics_layer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(affective_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(dependencies
  rclcpp
  nav2_costmap_2d
  pluginlib
  affective_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs)

add_library(proxemics_layer_core SHARED
  src/proxemic_layer.cpp
  src/proxemic_stamp.cpp)
target_include_directories(proxemics_layer_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(proxemics_layer_core ${dependencies})

pluginlib_export_plugin_description_file(nav2_costmap_2d proxemics_layer.xml)

install(TARGETS proxemics_layer_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES proxemics_layer.xml DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_stamp test/test_stamp.cpp)
  target_include_directories(test_stamp PUBLIC include)
  ament_target_dependencies(test_stamp nav2_costmap_2d)
  target_link_libraries(test_stamp proxemics_layer_core)
endif()

ament_export_include_directories(include)
ament_export_libraries(proxemics_layer_core)
ament_export_dependencies(${dependencies})
ament_package()
