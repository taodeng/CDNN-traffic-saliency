FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    sudo \
    bzip2 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda2-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 2.7 environment
RUN conda create -y --name py27 python=2.7 \
 && conda clean -ya

# Set py27 as default environment (py27 env linked outside conda, and py27 startup activation)
ENV CONDA_DEFAULT_ENV=py27
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN sudo ln -s /home/user/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
 && echo "source /etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate py27" >> ~/.bashrc

# CUDA 9.0-specific steps
RUN conda install -y -c pytorch \
    "pytorch=0.4.1=py27_cuda9.0.176_cudnn7.1.2_1" \
    cuda90 \
 && conda clean -ya

# Install OpenCV 3.4.0.14 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    libsm6 libxext6 libxrender-dev \
 && sudo rm -rf /var/lib/apt/lists/*

# Requirements
COPY requirements.txt /app/requirements.txt
RUN bash -c "source /etc/profile.d/conda.sh \
 && conda activate py27 \
 && pip install -r requirements.txt" \
 && rm /app/requirements.txt

# Default entry point
CMD ["bash"]
