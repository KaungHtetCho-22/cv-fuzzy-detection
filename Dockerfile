FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install system dependencies for OpenCV GUI (imshow, namedWindow, etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["bash"]
