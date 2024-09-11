FROM docker.io/nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

LABEL description="LLama Factory and Gpu envirment"

ARG TZ=Asia/Shanghai
ENV TZ ${TZ}

# change source
RUN cd /etc/apt/ && \
    mv sources.list sources.list.bak && \
    touch sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" >> sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> sources.list && \
    apt update && apt upgrade -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip git vim git-lfs wget ripgrep curl language-pack-zh-hans && \
    pip3 install --upgrade huggingface_hub modelscope jupyter TensorRT && \
    pip3 config set global.index-url https://pypi.doubanio.com/simple && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    cd /etc/apt/sources.list.d && \
    mv cuda-ubuntu2204-x86_64.list cuda-ubuntu2204-x86_64.list.bak && \
    echo "set encoding=utf-8" >> ~/.vimrc && \
    echo "set fileencodings=utf-8,gbk,gb2312,gb18030" >> ~/.vimrc && \
    echo "set termencoding=utf-8" >> ~/.vimrc

RUN mkdir -p /opt/env && \
    cd /opt/env && \
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory/ && \
    pip install -e ".[torch,metrics]"
