# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/catkin_ws/build

# Include any dependencies generated for this target.
include ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/depend.make

# Include the progress variables for this target.
include ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/progress.make

# Include the compile flags for this target's objects.
include ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/flags.make

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/flags.make
ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o: /root/catkin_ws/src/ros_astra_camera/test/test_wrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o"
	cd /root/catkin_ws/build/ros_astra_camera && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o -c /root/catkin_ws/src/ros_astra_camera/test/test_wrapper.cpp

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.i"
	cd /root/catkin_ws/build/ros_astra_camera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/catkin_ws/src/ros_astra_camera/test/test_wrapper.cpp > CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.i

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.s"
	cd /root/catkin_ws/build/ros_astra_camera && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/catkin_ws/src/ros_astra_camera/test/test_wrapper.cpp -o CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.s

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.requires:

.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.requires

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.provides: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.requires
	$(MAKE) -f ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/build.make ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.provides.build
.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.provides

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.provides.build: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o


# Object files for target astra_test_wrapper
astra_test_wrapper_OBJECTS = \
"CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o"

# External object files for target astra_test_wrapper
astra_test_wrapper_EXTERNAL_OBJECTS =

/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/build.make
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /root/catkin_ws/devel/lib/libastra_wrapper.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_system.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libpthread.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libcamera_info_manager.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libcamera_calibration_parsers.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libdynamic_reconfigure_config_init_mutex.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libimage_transport.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libmessage_filters.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libnodeletlib.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libbondcpp.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libclass_loader.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/libPocoFoundation.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libdl.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libroslib.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/librospack.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libpython2.7.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libroscpp.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/librosconsole.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/librosconsole_log4cxx.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/librosconsole_backend_interface.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libroscpp_serialization.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libxmlrpcpp.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/librostime.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /opt/ros/melodic/lib/libcpp_common.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_system.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: /usr/lib/x86_64-linux-gnu/libpthread.so
/root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper"
	cd /root/catkin_ws/build/ros_astra_camera && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/astra_test_wrapper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/build: /root/catkin_ws/devel/lib/astra_camera/astra_test_wrapper

.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/build

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/requires: ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/test/test_wrapper.cpp.o.requires

.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/requires

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/clean:
	cd /root/catkin_ws/build/ros_astra_camera && $(CMAKE_COMMAND) -P CMakeFiles/astra_test_wrapper.dir/cmake_clean.cmake
.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/clean

ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/depend:
	cd /root/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/catkin_ws/src /root/catkin_ws/src/ros_astra_camera /root/catkin_ws/build /root/catkin_ws/build/ros_astra_camera /root/catkin_ws/build/ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ros_astra_camera/CMakeFiles/astra_test_wrapper.dir/depend

