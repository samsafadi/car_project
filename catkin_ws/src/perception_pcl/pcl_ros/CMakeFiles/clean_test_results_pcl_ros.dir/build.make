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
CMAKE_BINARY_DIR = /root/catkin_ws/src

# Utility rule file for clean_test_results_pcl_ros.

# Include the progress variables for this target.
include perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/progress.make

perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros:
	cd /root/catkin_ws/src/perception_pcl/pcl_ros && /usr/bin/python2 /opt/ros/melodic/share/catkin/cmake/test/remove_test_results.py /root/catkin_ws/src/test_results/pcl_ros

clean_test_results_pcl_ros: perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros
clean_test_results_pcl_ros: perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/build.make

.PHONY : clean_test_results_pcl_ros

# Rule to build all files generated by this target.
perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/build: clean_test_results_pcl_ros

.PHONY : perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/build

perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/clean:
	cd /root/catkin_ws/src/perception_pcl/pcl_ros && $(CMAKE_COMMAND) -P CMakeFiles/clean_test_results_pcl_ros.dir/cmake_clean.cmake
.PHONY : perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/clean

perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/depend:
	cd /root/catkin_ws/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/catkin_ws/src /root/catkin_ws/src/perception_pcl/pcl_ros /root/catkin_ws/src /root/catkin_ws/src/perception_pcl/pcl_ros /root/catkin_ws/src/perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : perception_pcl/pcl_ros/CMakeFiles/clean_test_results_pcl_ros.dir/depend

