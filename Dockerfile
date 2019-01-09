# OS image
FROM ubuntu:16.04

# Install basic software
RUN \
  apt-get update -qq && \
  apt-get upgrade -qq && \
  apt-get install -qq build-essential && \
  apt-get install -qq software-properties-common && \
  apt-get install -qq byobu curl git htop man unzip vim wget

RUN apt-get install -qq python2.7 python-pip
RUN pip install --upgrade pip

# Install Boost 1.58.0
RUN wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz -q -O boost.tar.gz
RUN tar -zxf boost.tar.gz
RUN cd boost_1_58_0 && \
  ./bootstrap.sh && \
  ./b2 address-model=64 --with-system --with-filesystem --with-thread --prefix=/usr/local install

# Check installation
RUN gcc --version
RUN g++ --version
RUN pip --version

# Install Mitsuba dependencies
RUN pip install scons
RUN apt-get install -qq \
  build-essential libpng12-dev \
  libjpeg8-dev libilmbase-dev libxerces-c-dev libopenexr-dev \
  libglewmx-dev libeigen3-dev libfftw3-dev

# Install Mitsuba renderer
RUN git clone https://github.com/tatsy/mitsuba.git $HOME/mitsuba
RUN \
  cd $HOME/mitsuba && \
  cp build/config-linux-gcc.py config.py
RUN \
  cd $HOME/mitsuba && \
  scons -j4

# Setup ZSH
RUN apt-get install -qq zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true

# Environment path
ENV LD_LIBRARY_PATH /root/mitsuba/dist

# Run setting
CMD ["/bin/zsh"]
WORKDIR "/root/mitsuba"

