#!/bin/bash

### make
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=/usr/local/bin/python3
make

##install
cd ..
python install.py
