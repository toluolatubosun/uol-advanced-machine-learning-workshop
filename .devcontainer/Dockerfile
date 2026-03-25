FROM tensorflow/tensorflow:2.17.0-gpu

RUN pip install --upgrade pip

# Install PyTorch
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu126/torch_stable.html

# Install Python packages
RUN pip install ipykernel pillow vizdoom stable-baselines3 opencv-python swig "gymnasium[mujoco]"

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    libx11-6 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install stable-baselines3

RUN pip install "numpy<2"