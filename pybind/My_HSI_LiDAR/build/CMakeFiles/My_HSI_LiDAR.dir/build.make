# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build

# Include any dependencies generated for this target.
include CMakeFiles/My_HSI_LiDAR.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/My_HSI_LiDAR.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/My_HSI_LiDAR.dir/flags.make

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o: CMakeFiles/My_HSI_LiDAR.dir/flags.make
CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o -c /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/main.cpp

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/My_HSI_LiDAR.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/main.cpp > CMakeFiles/My_HSI_LiDAR.dir/main.cpp.i

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/My_HSI_LiDAR.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/main.cpp -o CMakeFiles/My_HSI_LiDAR.dir/main.cpp.s

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.requires

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.provides: CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/My_HSI_LiDAR.dir/build.make CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.provides

CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.provides.build: CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o


CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o: CMakeFiles/My_HSI_LiDAR.dir/flags.make
CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o: ../ndarray_converter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o -c /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/ndarray_converter.cpp

CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/ndarray_converter.cpp > CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.i

CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/ndarray_converter.cpp -o CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.s

CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.requires:

.PHONY : CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.requires

CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.provides: CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.requires
	$(MAKE) -f CMakeFiles/My_HSI_LiDAR.dir/build.make CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.provides.build
.PHONY : CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.provides

CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.provides.build: CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o


# Object files for target My_HSI_LiDAR
My_HSI_LiDAR_OBJECTS = \
"CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o" \
"CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o"

# External object files for target My_HSI_LiDAR
My_HSI_LiDAR_EXTERNAL_OBJECTS =

My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/My_HSI_LiDAR.dir/build.make
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libGL.so
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_gapi.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_stitching.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_aruco.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_bgsegm.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_bioinspired.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ccalib.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_cvv.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn_objdetect.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn_superres.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dpm.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_face.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_freetype.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_fuzzy.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_hdf.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_hfs.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_img_hash.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_line_descriptor.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_quality.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_reg.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_rgbd.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_saliency.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_sfm.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_stereo.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_structured_light.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_superres.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_surface_matching.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_tracking.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_videostab.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xfeatures2d.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xobjdetect.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_xphoto.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_highgui.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_shape.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_datasets.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_plot.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_text.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_dnn.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ml.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_phase_unwrapping.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_optflow.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_ximgproc.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_video.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_videoio.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgcodecs.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_objdetect.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_calib3d.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_features2d.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_flann.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_photo.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_imgproc.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: /usr/local/lib/libopencv_core.so.4.1.2
My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/My_HSI_LiDAR.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/My_HSI_LiDAR.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/My_HSI_LiDAR.dir/build: My_HSI_LiDAR.cpython-36m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/My_HSI_LiDAR.dir/build

CMakeFiles/My_HSI_LiDAR.dir/requires: CMakeFiles/My_HSI_LiDAR.dir/main.cpp.o.requires
CMakeFiles/My_HSI_LiDAR.dir/requires: CMakeFiles/My_HSI_LiDAR.dir/ndarray_converter.cpp.o.requires

.PHONY : CMakeFiles/My_HSI_LiDAR.dir/requires

CMakeFiles/My_HSI_LiDAR.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/My_HSI_LiDAR.dir/cmake_clean.cmake
.PHONY : CMakeFiles/My_HSI_LiDAR.dir/clean

CMakeFiles/My_HSI_LiDAR.dir/depend:
	cd /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build /home/jxd/Documents/jxd/code/pybind/My_HSI_LiDAR/build/CMakeFiles/My_HSI_LiDAR.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/My_HSI_LiDAR.dir/depend

