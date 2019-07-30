cvVersion="3.4.4"
mkdir installation
mkdir installation/OpenCV-"$cvVersion"
cwd=$(pwd)

cp -r ./python-opencv-cuda/c++/pythoncuda/ ./opencv_contrib/modules

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${cwd}"/installation/OpenCV-"${cvVersion} \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D WITH_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.5/dist-packages \
    -D BUILD_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D BUILD_opencv_python2=OFF \
..
make -j4
make install
ldconfig

