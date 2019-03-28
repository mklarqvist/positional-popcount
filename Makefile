###################################################################
# Copyright (c) 2019
# Author(s): Marcus D. R. Klarqvist and Daniel Lemire
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

OPTFLAGS  := -O3 -march=native
CFLAGS     = -std=c99 $(OPTFLAGS) $(DEBUG_FLAGS)
CPPFLAGS   = -std=c++0x $(OPTFLAGS) $(DEBUG_FLAGS)
CPP_SOURCE = main.cpp
C_SOURCE   = fast_flagstats.c example.c
OBJECTS    = $(CPP_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)

# Default target
all: fast_flag_stats

# Generic rules
%.o: %.c
	$(CC) $(CFLAGS)-c -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS)-c -o $@ $<

fast_flag_stats: fast_flagstats.o main.o
	$(CXX) $(CPPFLAGS) fast_flagstats.c main.cpp -o fast_flag_stats

itest: instrumented_benchmark
	$(CXX) --version
	sudo ./instrumented_benchmark

instrumented_benchmark: benchmark/linux/instrumented_benchmark.cpp benchmark/linux/linux-perf-events.h fast_flagstats.h fast_flagstats.c
	$(CXX) $(CPPFLAGS) fast_flagstats.c  benchmark/linux/instrumented_benchmark.cpp -I. -Ibenchmark/linux -o instrumented_benchmark

example: fast_flagstats.o example.o
	$(CC) $(CFLAGS) fast_flagstats.c example.c -o example

test: fast_flag_stats
	$(CXX) --version
	./fast_flag_stats

clean:
	rm -f $(OBJECTS)
	rm -f fast_flag_stats example instrumented_benchmark

.PHONY: all clean test itest
