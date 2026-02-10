# Compile
cd ~/neural-optics
rm -rf build_cmake
mkdir build_cmake && cd build_cmake
cmake -G Ninja ..
ninja
# Directories
## py
### scripts
1. generate-scene-datasets
2. generate-target-datasets
