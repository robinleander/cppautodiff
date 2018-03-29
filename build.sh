#!/bin/sh

cmake . -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
make -j4

