cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} src/main.cpp src/cl/clion_defines.cl src/cl/aplusb.cl)
target_link_libraries(${PROJECT_NAME} libclew libutils)
