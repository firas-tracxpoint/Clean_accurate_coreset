cmake_minimum_required(VERSION 2.8.4)
project(for_danny)
set(sample_dir ${CMAKE_CURRENT_SOURCE_DIR}/)
set(MKLROOT "/home/firas/intel/compilers_and_libraries_2020.1.217/linux/mkl/")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -m64 -I${MKLROOT}/include")
find_package( OpenCV REQUIRED )
set(sample_pnplib
        ${sample_dir}pnp_math.cpp
        ${sample_dir}Coresets.cpp
        )
ocv_include_modules_recurse(${OPENCV_CPP_SAMPLES_REQUIRED_DEPS})
add_executable(for_danny main.cpp ${sample_pnplib} Coresets.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( for_danny ${OpenCV_LIBS} -Wl,--start-group
                    ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a
                    ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a
                    ${MKLROOT}/lib/intel64/libmkl_core.a
                    -Wl,--end-group -ltbb -lstdc++ -lpthread -lm -ldl)

