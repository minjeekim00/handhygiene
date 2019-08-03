cvVersion="3.4.4"
mkdir installation
mkdir installation/OpenCV-"$cvVersion"
cwd=$(pwd)

apt -y update
apt -y upgrade
apt -y remove x264 libx264-dev

## Install dependencies
apt -y install build-essential checkinstall cmake pkg-config yasm
apt -y install git gfortran
apt -y install libjpeg8-dev libjasper-dev libpng12-dev
apt -y install libtiff5-dev
apt -y install libtiff-dev
apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd $cwd

apt -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
apt -y install libgtk2.0-dev libtbb-dev qt5-default
apt -y install libopenblas-dev liblapack-dev libatlas-base-dev
apt -y install libfaac-dev libmp3lame-dev libtheora-dev
apt -y install libvorbis-dev libxvidcore-dev
apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
apt -y install libavresample-dev
apt -y install x264 v4l-utils

# Optional dependencies
apt -y install libprotobuf-dev protobuf-compiler
apt -y install libgoogle-glog-dev libgflags-dev
apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Install Python Libraries
apt -y install python3-dev python3-pip

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $cvVersion
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $cvVersion
cd ..


apt-get install qt-sdk -y
python3 -m pip install numpy
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
