# C++ Extensions for Graph Sampler in Python

The sampler of the graph + other misc functions used in our implementation.

The graph is assumed to have this format
- node_types: ...
- end_points: ...
- ind_ptr: ...
- node_ids: ...


# Install
For windows users:

```bash
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ..
```
Open GraphSampler.sln and use VS 2015 to build, then
```bash
cd ..
python install.py
```

For unix users:

Firstly, install the https://github.com/sparsehash/sparsehash according to the guidelines.
```bash
git clone https://github.com/sparsehash/sparsehash
cd sparsehash
./configure
make
sudo make install
```
It will install the header files into some paths like `/usr/local/include`.

Then, use cmake to install the package

```bash
mkdir build
cd build
cmake ..
make
cd ..
python install.py
```
