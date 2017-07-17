
add_definitions(-DTHIS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(reconstruction Reconstruct.cpp)
target_link_libraries(reconstruction
  openMVG_image
  openMVG_features
  openMVG_matching
  openMVG_system
  openMVG_sfm
  stlplus
  vlsift)

set_property(TARGET reconstruction)
