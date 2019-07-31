version="3.2"
# ubuntu >= 18.04 LTS
#sudo apt-get install -y python3-dev pkg-config
#sudo apt-get install -y libavformat-dev
#sudo apt-get install -y libavcodec-dev libavdevice-dev \
#    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev

# ubuntu < 18.04 LTS
sudo apt install -y autoconf automake build-essential cmake \
    libass-dev libfreetype6-dev libjpeg-dev libtheora-dev \
    libtool libvorbis-dev libx264-dev pkg-config wget yasm zlib1g-dev

# unzip the ffmpeg source code and install
wget http://ffmpeg.org/releases/ffmpeg-${version}.tar.bz2
tar -xjvf ffmpeg-${version}.tar.bz2

cd ffmpeg-${version}
# ubuntu 18.04
#./configure --prefix=/usr/local/ffmpeg --enable-gpl --enable-version3 --enable-nonfree --enable-postproc --enable-pthreads --enable-libfdk_aac --enable-libmp3lame --enable-libtheora --enable-libx264 --enable-libxvid --enable-libxcb --enable-libvorbis
# ubuntu < 18.04
./configure --disable-static --enable-shared --disable-doc

sudo make
sudo make install

# configure the PATH param
echo "# Add FFMpeg bin & library paths:" >> ~/.bashrc
echo "export PATH=/usr/local/ffmpeg/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

## You can also install from a PPA
# sudo add-apt-repository ppa:kirillshkrogalev/ffmpeg-next
# sudo apt-get update
# sudo apt-get install ffmpeg
