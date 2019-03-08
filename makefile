###################################################################
# Copyright (c) 2019 Marcus D. R. Klarqvist
# Author(s): Marcus D. R. Klarqvist
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
###################################################################

OPTFLAGS := -O3 -march=native -mtune=native
CXXFLAGS      = -std=c++0x $(OPTFLAGS) $(DEBUG_FLAGS)
CFLAGS        = -std=c99   $(OPTFLAGS) $(DEBUG_FLAGS)
CFLAGS_VENDOR = -std=c99   $(OPTFLAGS)

LIBS := 
CXX_SOURCE = $(wildcard *.cpp)
C_SOURCE = 

OBJECTS  = $(CXX_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)
CPP_DEPS = $(CXX_SOURCE:.cpp=.d) $(C_SOURCE:.c=.d)

# Default target
all: fast_flag_stats

# Generic rules
%.o: %.cpp
	g++ $(CXXFLAGS) $(INCLUDE_PATH) -c -o $@ $<

fast_flag_stats: $(OBJECTS)
	g++ $(BINARY_RPATHS) $(LIBRARY_PATHS) -pthread $(OBJECTS) $(LIBS) -o fast_flag_stats

clean:
	rm -f $(OBJECTS) $(CPP_DEPS)
	rm -f fast_flag_stats

.PHONY: all clean