cmake_minimum_required(VERSION 3.0.2)
project(proxemics_layer_ros1)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  costmap_2d
  pluginlib
  affective_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES proxemics_layer_ros1
  CATKIN_DEPENDS roscpp costmap_2d pluginlib affective_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(proxemics_layer_ros1
  src/proxemic_layer.cpp
)
target_link_libraries(proxemics_layer_ros1 ${catkin_LIBRARIES})
add_dependencies(proxemics_layer_ros1 ${catkin_EXPORTED_TARGETS})

install(TARGETS proxemics_layer_ros1
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(FILES proxemics_layer_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
