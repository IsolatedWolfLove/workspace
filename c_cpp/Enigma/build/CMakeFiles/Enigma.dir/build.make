# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ccy/workspace/c_cpp/Enigma

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ccy/workspace/c_cpp/Enigma/build

# Include any dependencies generated for this target.
include CMakeFiles/Enigma.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Enigma.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Enigma.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Enigma.dir/flags.make

CMakeFiles/Enigma.dir/decode.cpp.o: CMakeFiles/Enigma.dir/flags.make
CMakeFiles/Enigma.dir/decode.cpp.o: /home/ccy/workspace/c_cpp/Enigma/decode.cpp
CMakeFiles/Enigma.dir/decode.cpp.o: CMakeFiles/Enigma.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ccy/workspace/c_cpp/Enigma/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Enigma.dir/decode.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Enigma.dir/decode.cpp.o -MF CMakeFiles/Enigma.dir/decode.cpp.o.d -o CMakeFiles/Enigma.dir/decode.cpp.o -c /home/ccy/workspace/c_cpp/Enigma/decode.cpp

CMakeFiles/Enigma.dir/decode.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Enigma.dir/decode.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ccy/workspace/c_cpp/Enigma/decode.cpp > CMakeFiles/Enigma.dir/decode.cpp.i

CMakeFiles/Enigma.dir/decode.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Enigma.dir/decode.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ccy/workspace/c_cpp/Enigma/decode.cpp -o CMakeFiles/Enigma.dir/decode.cpp.s

# Object files for target Enigma
Enigma_OBJECTS = \
"CMakeFiles/Enigma.dir/decode.cpp.o"

# External object files for target Enigma
Enigma_EXTERNAL_OBJECTS =

Enigma: CMakeFiles/Enigma.dir/decode.cpp.o
Enigma: CMakeFiles/Enigma.dir/build.make
Enigma: CMakeFiles/Enigma.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ccy/workspace/c_cpp/Enigma/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Enigma"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Enigma.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Enigma.dir/build: Enigma
.PHONY : CMakeFiles/Enigma.dir/build

CMakeFiles/Enigma.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Enigma.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Enigma.dir/clean

CMakeFiles/Enigma.dir/depend:
	cd /home/ccy/workspace/c_cpp/Enigma/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ccy/workspace/c_cpp/Enigma /home/ccy/workspace/c_cpp/Enigma /home/ccy/workspace/c_cpp/Enigma/build /home/ccy/workspace/c_cpp/Enigma/build /home/ccy/workspace/c_cpp/Enigma/build/CMakeFiles/Enigma.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Enigma.dir/depend

