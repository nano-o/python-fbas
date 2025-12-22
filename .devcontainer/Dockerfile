FROM debian:stable

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libzmq3-dev \
    zlib1g-dev \
    curl \
    git \
    python3-pip \
    python3-venv \
    ncurses-term \
    locales \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

RUN useradd -m -s /bin/bash vscode
USER vscode

RUN python3 -m venv /home/vscode/.venv
ENV PATH="/home/vscode/.venv/bin:$PATH"

WORKDIR /workspaces/python-fbas
