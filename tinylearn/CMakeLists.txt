cmake_minimum_required(VERSION 3.15)
project(TinyLearn VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Find dependencies
find_package(OpenCV 4 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================
# MNIST Training Executable
# ============================================================
add_executable(tinylearn_mnist
    src/mnist_train.cpp
)

target_link_libraries(tinylearn_mnist
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    pthread
)

target_compile_options(tinylearn_mnist PRIVATE
    -fopenmp
)

# ============================================================
# CIFAR-100 Training Executable
# ============================================================
add_executable(tinylearn_cifar100
    src/cifar100_train.cpp
    src/tensor.cpp
    src/layers.cpp
    src/dataset.cpp
)

target_link_libraries(tinylearn_cifar100
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    pthread
)

target_compile_options(tinylearn_cifar100 PRIVATE
    -fopenmp
)

# ============================================================
# Python Bindings (pybind11 module)
# ============================================================
pybind11_add_module(tinylearn
    src/bindings.cpp
)

target_link_libraries(tinylearn PRIVATE
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_compile_options(tinylearn PRIVATE
    -fopenmp
    -O3
    -fPIC
)

# ============================================================
# Installation
# ============================================================
install(TARGETS tinylearn_mnist tinylearn_cifar100
    RUNTIME DESTINATION bin
)

install(TARGETS tinylearn
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

install(DIRECTORY include/tinylearn
    DESTINATION include
)

# ============================================================
# Testing (optional — requires Catch2)
# ============================================================
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    find_package(Catch2 3 QUIET)
    
    if(Catch2_FOUND)
        add_executable(tinylearn_tests
            tests/test_tensor.cpp
            tests/test_model.cpp
            tests/test_data_loader.cpp
        )
        
        target_link_libraries(tinylearn_tests PRIVATE
            Catch2::Catch2WithMain
            ${OpenCV_LIBS}
            OpenMP::OpenMP_CXX
        )
        
        include(CTest)
        include(Catch)
        catch_discover_tests(tinylearn_tests)
    else()
        message(WARNING "Catch2 not found. Skipping tests.")
    endif()
endif()

# ============================================================
# Print configuration summary
# ============================================================
message(STATUS "===========================================")
message(STATUS "TinyLearn Build Configuration")
message(STATUS "===========================================")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags:        ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenCV version:   ${OpenCV_VERSION}")
message(STATUS "Python version:   ${Python3_VERSION}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build tests:      ${BUILD_TESTS}")
message(STATUS "===========================================")
