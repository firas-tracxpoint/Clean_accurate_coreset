# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/114/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/114/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/firas/CLionProjects/for_danny

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/firas/CLionProjects/for_danny/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/for_danny.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/for_danny.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/for_danny.dir/flags.make

CMakeFiles/for_danny.dir/main.cpp.o: CMakeFiles/for_danny.dir/flags.make
CMakeFiles/for_danny.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firas/CLionProjects/for_danny/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/for_danny.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/for_danny.dir/main.cpp.o -c /home/firas/CLionProjects/for_danny/main.cpp

CMakeFiles/for_danny.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/for_danny.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firas/CLionProjects/for_danny/main.cpp > CMakeFiles/for_danny.dir/main.cpp.i

CMakeFiles/for_danny.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/for_danny.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firas/CLionProjects/for_danny/main.cpp -o CMakeFiles/for_danny.dir/main.cpp.s

CMakeFiles/for_danny.dir/pnp_math.cpp.o: CMakeFiles/for_danny.dir/flags.make
CMakeFiles/for_danny.dir/pnp_math.cpp.o: ../pnp_math.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firas/CLionProjects/for_danny/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/for_danny.dir/pnp_math.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/for_danny.dir/pnp_math.cpp.o -c /home/firas/CLionProjects/for_danny/pnp_math.cpp

CMakeFiles/for_danny.dir/pnp_math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/for_danny.dir/pnp_math.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firas/CLionProjects/for_danny/pnp_math.cpp > CMakeFiles/for_danny.dir/pnp_math.cpp.i

CMakeFiles/for_danny.dir/pnp_math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/for_danny.dir/pnp_math.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firas/CLionProjects/for_danny/pnp_math.cpp -o CMakeFiles/for_danny.dir/pnp_math.cpp.s

CMakeFiles/for_danny.dir/Coresets.cpp.o: CMakeFiles/for_danny.dir/flags.make
CMakeFiles/for_danny.dir/Coresets.cpp.o: ../Coresets.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/firas/CLionProjects/for_danny/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/for_danny.dir/Coresets.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/for_danny.dir/Coresets.cpp.o -c /home/firas/CLionProjects/for_danny/Coresets.cpp

CMakeFiles/for_danny.dir/Coresets.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/for_danny.dir/Coresets.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/firas/CLionProjects/for_danny/Coresets.cpp > CMakeFiles/for_danny.dir/Coresets.cpp.i

CMakeFiles/for_danny.dir/Coresets.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/for_danny.dir/Coresets.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/firas/CLionProjects/for_danny/Coresets.cpp -o CMakeFiles/for_danny.dir/Coresets.cpp.s

# Object files for target for_danny
for_danny_OBJECTS = \
"CMakeFiles/for_danny.dir/main.cpp.o" \
"CMakeFiles/for_danny.dir/pnp_math.cpp.o" \
"CMakeFiles/for_danny.dir/Coresets.cpp.o"

# External object files for target for_danny
for_danny_EXTERNAL_OBJECTS =

for_danny: CMakeFiles/for_danny.dir/main.cpp.o
for_danny: CMakeFiles/for_danny.dir/pnp_math.cpp.o
for_danny: CMakeFiles/for_danny.dir/Coresets.cpp.o
for_danny: CMakeFiles/for_danny.dir/build.make
for_danny: /usr/local/lib/libopencv_stitching.so.3.4.9
for_danny: /usr/local/lib/libopencv_superres.so.3.4.9
for_danny: /usr/local/lib/libopencv_videostab.so.3.4.9
for_danny: /usr/local/lib/libopencv_aruco.so.3.4.9
for_danny: /usr/local/lib/libopencv_bgsegm.so.3.4.9
for_danny: /usr/local/lib/libopencv_bioinspired.so.3.4.9
for_danny: /usr/local/lib/libopencv_ccalib.so.3.4.9
for_danny: /usr/local/lib/libopencv_cvv.so.3.4.9
for_danny: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.9
for_danny: /usr/local/lib/libopencv_dpm.so.3.4.9
for_danny: /usr/local/lib/libopencv_face.so.3.4.9
for_danny: /usr/local/lib/libopencv_freetype.so.3.4.9
for_danny: /usr/local/lib/libopencv_fuzzy.so.3.4.9
for_danny: /usr/local/lib/libopencv_hdf.so.3.4.9
for_danny: /usr/local/lib/libopencv_hfs.so.3.4.9
for_danny: /usr/local/lib/libopencv_img_hash.so.3.4.9
for_danny: /usr/local/lib/libopencv_line_descriptor.so.3.4.9
for_danny: /usr/local/lib/libopencv_optflow.so.3.4.9
for_danny: /usr/local/lib/libopencv_reg.so.3.4.9
for_danny: /usr/local/lib/libopencv_rgbd.so.3.4.9
for_danny: /usr/local/lib/libopencv_saliency.so.3.4.9
for_danny: /usr/local/lib/libopencv_sfm.so.3.4.9
for_danny: /usr/local/lib/libopencv_stereo.so.3.4.9
for_danny: /usr/local/lib/libopencv_structured_light.so.3.4.9
for_danny: /usr/local/lib/libopencv_surface_matching.so.3.4.9
for_danny: /usr/local/lib/libopencv_tracking.so.3.4.9
for_danny: /usr/local/lib/libopencv_xfeatures2d.so.3.4.9
for_danny: /usr/local/lib/libopencv_ximgproc.so.3.4.9
for_danny: /usr/local/lib/libopencv_xobjdetect.so.3.4.9
for_danny: /usr/local/lib/libopencv_xphoto.so.3.4.9
for_danny: /home/firas/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64/libmkl_intel_ilp64.a
for_danny: /home/firas/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64/libmkl_tbb_thread.a
for_danny: /home/firas/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64/libmkl_core.a
for_danny: /usr/local/lib/libopencv_highgui.so.3.4.9
for_danny: /usr/local/lib/libopencv_videoio.so.3.4.9
for_danny: /usr/local/lib/libopencv_shape.so.3.4.9
for_danny: /usr/local/lib/libopencv_viz.so.3.4.9
for_danny: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.9
for_danny: /usr/local/lib/libopencv_video.so.3.4.9
for_danny: /usr/local/lib/libopencv_datasets.so.3.4.9
for_danny: /usr/local/lib/libopencv_plot.so.3.4.9
for_danny: /usr/local/lib/libopencv_text.so.3.4.9
for_danny: /usr/local/lib/libopencv_dnn.so.3.4.9
for_danny: /usr/local/lib/libopencv_ml.so.3.4.9
for_danny: /usr/local/lib/libopencv_imgcodecs.so.3.4.9
for_danny: /usr/local/lib/libopencv_objdetect.so.3.4.9
for_danny: /usr/local/lib/libopencv_calib3d.so.3.4.9
for_danny: /usr/local/lib/libopencv_features2d.so.3.4.9
for_danny: /usr/local/lib/libopencv_flann.so.3.4.9
for_danny: /usr/local/lib/libopencv_photo.so.3.4.9
for_danny: /usr/local/lib/libopencv_imgproc.so.3.4.9
for_danny: /usr/local/lib/libopencv_core.so.3.4.9
for_danny: CMakeFiles/for_danny.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/firas/CLionProjects/for_danny/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable for_danny"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/for_danny.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/for_danny.dir/build: for_danny

.PHONY : CMakeFiles/for_danny.dir/build

CMakeFiles/for_danny.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/for_danny.dir/cmake_clean.cmake
.PHONY : CMakeFiles/for_danny.dir/clean

CMakeFiles/for_danny.dir/depend:
	cd /home/firas/CLionProjects/for_danny/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/firas/CLionProjects/for_danny /home/firas/CLionProjects/for_danny /home/firas/CLionProjects/for_danny/cmake-build-debug /home/firas/CLionProjects/for_danny/cmake-build-debug /home/firas/CLionProjects/for_danny/cmake-build-debug/CMakeFiles/for_danny.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/for_danny.dir/depend

