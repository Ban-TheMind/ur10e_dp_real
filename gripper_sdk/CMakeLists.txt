cmake_minimum_required(VERSION 3.16)
project(robotiq_sdk_bridge LANGUAGES C CXX)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GRIPPERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GRIPPERS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GRIPPERS_BUILD_FAKE OFF CACHE BOOL "" FORCE)
set(GRIPPERS_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
# Build the SDK's transport dependency from an unmodified pinned release.
include(ExternalProject)
find_program(SERIALPORT_MAKE make)
if(NOT SERIALPORT_MAKE)
  message(FATAL_ERROR "make is required to build the official SDK serial transport")
endif()
set(SERIALPORT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/serialport")
file(MAKE_DIRECTORY "${SERIALPORT_PREFIX}/include")
set(EXTRACT_TIMESTAMP_ARGS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set(EXTRACT_TIMESTAMP_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()
ExternalProject_Add(serialport_build
  URL "${CMAKE_CURRENT_SOURCE_DIR}/vendor/libserialport/libserialport-0.1.2.tar.gz"
  ${EXTRACT_TIMESTAMP_ARGS}
  URL_HASH SHA256=5deb92b5ca72c0347b07b786848350deca2dcfd975ce613b8e0e1d947a4b4ca9
  CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${SERIALPORT_PREFIX}
    --disable-shared --enable-static
    "CC=${CMAKE_C_COMPILER}" "CFLAGS=-O2 -fPIC"
  BUILD_COMMAND ${SERIALPORT_MAKE}
  INSTALL_COMMAND ${SERIALPORT_MAKE} install
  BUILD_BYPRODUCTS "${SERIALPORT_PREFIX}/lib/libserialport.a")
add_library(serialport::serialport STATIC IMPORTED GLOBAL)
set_target_properties(serialport::serialport PROPERTIES
  IMPORTED_LOCATION "${SERIALPORT_PREFIX}/lib/libserialport.a"
  INTERFACE_INCLUDE_DIRECTORIES "${SERIALPORT_PREFIX}/include")
add_dependencies(serialport::serialport serialport_build)
# The full pinned SDK source and licenses travel as one archive.
set(SDK_ROOT "${CMAKE_CURRENT_BINARY_DIR}/sdk_source")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/vendor/robotiq_sdk.tar.gz" SDK_HASH)
if(NOT SDK_HASH STREQUAL "1aa0ab8c8bc10d6c37f6d838a1d086af9bb63bf273c33daa42fd0eccf01f7cf5")
  message(FATAL_ERROR "Robotiq SDK source archive checksum mismatch")
endif()
file(MAKE_DIRECTORY "${SDK_ROOT}")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xzf
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/robotiq_sdk.tar.gz"
  WORKING_DIRECTORY "${SDK_ROOT}" RESULT_VARIABLE SDK_EXTRACT_RESULT)
if(NOT SDK_EXTRACT_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to extract Robotiq SDK")
endif()
add_subdirectory("${SDK_ROOT}/robotiq/sdk_cpp" "${CMAKE_CURRENT_BINARY_DIR}/sdk" EXCLUDE_FROM_ALL)
add_executable(robotiq_sdk_command src/main.cpp)
target_link_libraries(robotiq_sdk_command PRIVATE Robotiq::grippers)
install(TARGETS robotiq_sdk_command DESTINATION lib/${PROJECT_NAME})
if(BUILD_TESTING)
  enable_testing()
  add_executable(test_bridge test/test_bridge.cpp)
  target_include_directories(test_bridge PRIVATE src "${SDK_ROOT}/robotiq/sdk_cpp/tests")
  target_link_libraries(test_bridge PRIVATE Robotiq::grippers)
  add_test(NAME robotiq_bridge_policy COMMAND test_bridge)
  add_executable(test_move_readiness test/test_move_readiness.cpp)
  target_include_directories(test_move_readiness PRIVATE src)
  target_link_libraries(test_move_readiness PRIVATE Robotiq::grippers)
  add_test(NAME robotiq_move_readiness COMMAND test_move_readiness)
endif()
