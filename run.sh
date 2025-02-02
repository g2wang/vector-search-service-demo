#!/bin/bash

# export LDFLAGS="-L/usr/local/opt/llvm@17/lib/c++ -Wl,-rpath,/usr/local/opt/llvm@17/lib/c++"
# export LDFLAGS="-L/usr/local/opt/llvm@17/lib"
# export CPPFLAGS="-I/usr/local/opt/llvm@17/include"
# LIBTORCH=/Users/guangdewang/pytorch-v2.6.0
export LIBTORCH=/usr/local/Cellar/pytorch/2.5.1_4
# export MACOSX_DEPLOYMENT_TARGET=15.2
# export CC=$(which clang)
# export CXX=$(which clang++)
cargo run
