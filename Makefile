###################################################################
# Copyright (c) 2019
# Author(s): Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire
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
C_SOURCE   = pospopcnt.c example.c
OBJECTS    = $(CPP_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)

# Default target
all: bench

# Generic rules
%.o: %.c
	$(CC) $(CFLAGS)-c -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS)-c -o $@ $<

bench: pospopcnt.o main.o
	$(CXX) $(CPPFLAGS) -ffast-math pospopcnt.c main.cpp -o bench

itest: instrumented_benchmark
	$(CXX) --version
	./instrumented_benchmark

instrumented_benchmark: benchmark/linux/instrumented_benchmark.cpp benchmark/linux/linux-perf-events.h pospopcnt.h pospopcnt.c
	$(CXX) $(CPPFLAGS) pospopcnt.c  benchmark/linux/instrumented_benchmark.cpp -I. -Ibenchmark/linux -o instrumented_benchmark

example: pospopcnt.o example.o
	$(CC) $(CFLAGS) pospopcnt.c example.c -o example

test: bench
	$(CXX) --version
	./bench

clean:
	rm -f $(OBJECTS)
	rm -f bench example instrumented_benchmark

.PHONY: all clean test itest
