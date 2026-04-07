FROM python:3.7

LABEL version="0.2.0"
LABEL maintainer="OpenMined"

COPY . /pyvertical
WORKDIR /pyvertical

# ---------------------------------------------------------
# 1. 【关键修改】强制切换到阿里云 Debian 源
# 这样能跑满国内带宽，通常只需 20-40 秒
# ---------------------------------------------------------
RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libsrtp2-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 2. 配置 PIP 源 (保持清华源)
# ---------------------------------------------------------
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf && \
    echo "trusted-host = mirrors.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf

# ---------------------------------------------------------
# 3. 安装 Python 依赖
# ---------------------------------------------------------
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install jupyterlab

# Expose port for jupyter lab
EXPOSE 8888

# Enter into jupyter lab
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
