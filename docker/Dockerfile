FROM rocm/tensorflow:rocm4.0-tf2.3-dev
MAINTAINER Ankur Deshwal <a.deshwal@samung.com>

# Default env setup to pass through damn proxy..
USER root
RUN mkdir -p /usr/share/ca-certificates/extra
RUN mkdir -p /tmp/files/

COPY sudo_1.8.16-0ubuntu1.9_amd64.deb /tmp/files/
COPY SMRC.crt /usr/share/ca-certificates/extra/
COPY ca-certificates_20170717~16.04.1_all.deb /tmp/files/

RUN dpkg -i /tmp/files/sudo*
RUN sed -i '/# Host alias /iDefaults\tenv_keep="http_proxy https_proxy ftp_proxy"' /etc/sudoers
RUN sed -i 's/archive.ubuntu.com/kr.archive.ubuntu.com/g' /etc/apt/sources.list

#RUN export http_proxy="http://75.12.251.5:8080"
#RUN export https_proxy="http://75.12.251.5:8080"
#ENV http_proxy http://75.12.251.5:8080
#ENV https_proxy http://75.12.251.5:8080
#ENV ftp_proxy http://75.12.251.5:8080
#ENV HTTP_PROXY http://75.12.251.5:8080
#ENV HTTPS_PROXY http://75.12.251.5:8080

RUN dpkg --add-architecture i386 && \
    dpkg -i /tmp/files/ca-cert*
RUN ls -al /etc/ssl/certs/SMRC.pem && \
    cat /etc/ca-certificates.conf | grep SMRC

RUN curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

RUN apt-get update

RUN apt-get --yes --allow-unauthenticated install openssl

RUN apt-get update

# Build Requirements
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends
RUN apt-get update
RUN apt-get --yes --allow-unauthenticated install sudo
RUN apt-get --yes --allow-unauthenticated install wget
RUN apt-get --yes --allow-unauthenticated install kmod

# Add rocm apt repository
RUN wget -qO - http://repo.radeon.com/rocm/apt/4.0/rocm.gpg.key | sudo apt-key add -
RUN echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.0/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

RUN apt-get update
# Instal hipcc 
#RUN apt-get install --yes --allow-unauthenticated rocm-libs hipcub miopen-hip
#RUN pip3 install --user tensorflow-rocm --upgrade

# Developement Requirments
RUN apt-get --yes --allow-unauthenticated install build-essential
RUN apt-get --yes --allow-unauthenticated install git
RUN apt-get --yes --allow-unauthenticated install vim
RUN apt-get --yes --allow-unauthenticated install byobu
RUN apt-get --yes --allow-unauthenticated install bash-completion
RUN apt-get --yes --allow-unauthenticated install clang-format-3.9
RUN apt-get --yes --allow-unauthenticated install cmake
RUN apt-get --yes --allow-unauthenticated install libpci-dev
RUN apt-get --yes --allow-unauthenticated install libsqlite3-dev
RUN apt-get --yes --allow-unauthenticated install libboost-all-dev

# PIM Library requirements
RUN apt-get --yes --allow-unauthenticated install libgoogle-glog-dev
RUN apt-get --yes --allow-unauthenticated install half
RUN apt-get update

ENV QWORKSPACE=/home/user/pim-workspace
ENV ROCM_PATH=/opt/rocm-4.0.0
ENV HIP_PATH=${ROCM_PATH}/hip
ENV HCC_HOME=${ROCM_PATH}/hcc
ENV HIP_PLATFORM=hcc
ENV PATH=${ROCM_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH=${ROCM_PATH}/hsa/lib:${ROCM_PATH}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV GOPATH=/home/user/go
ENV PATH=$GOPATH/bin:/usr/local/go/bin:$PATH

RUN groupadd -g 9000 developer

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
COPY ./bashrc /tmp/bashrc

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash", "-l"]
