cmake_minimum_required(VERSION 3.10)
project(assignment_1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(OpenMP)

find_package(OpenCV REQUIRED)

add_executable(train main.cpp)
target_link_libraries(train PRIVATE ${OpenCV_LIBS} Threads::Threads)
if(OpenMP_CXX_FOUND)
    target_link_libraries(train PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)

    execute_process(COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --cmake
                    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(WARNING "pybind11 not found via python module. Please install it (pip install pybind11).")
    endif()
endif()

if(pybind11_FOUND)
    pybind11_add_module(assignment_1 bindings.cpp)
    target_link_libraries(assignment_1 PRIVATE ${OpenCV_LIBS} Threads::Threads)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(assignment_1 PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    message(STATUS "Skipping python bindings build due to missing pybind11")
endif()