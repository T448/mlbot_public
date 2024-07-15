FROM python:3.11.7-slim-bullseye

ARG USERNAME=pyuser
ARG USER_DIR=/home/$USERNAME
ARG WORK_DIR=$USER_DIR/project
RUN adduser ${USERNAME}

USER root

RUN chown -R ${USERNAME}:${USERNAME} ${USER_DIR} && chmod -R 755 ${USER_DIR}
RUN mkdir ${WORK_DIR} && chown -R ${USERNAME}:${USERNAME} ${WORK_DIR} && chmod -R 755 ${WORK_DIR}

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    git \
    vim \
    tig \
    wget \
    unzip \
    curl \
    default-jre \
    libgomp1 \
    graphviz \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev  \
    xz-utils \
    tk-dev  \
    libxml2-dev  \
    libxmlsec1-dev \
    libffi-dev  \
    liblzma-dev
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    eval "$(pyenv init -)"

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# 不要なら消す(oh-my-posh)
RUN wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64 -O /usr/local/bin/oh-my-posh
RUN chmod +x /usr/local/bin/oh-my-posh

USER ${USERNAME}
RUN pip install -U pip
COPY ./requirements.txt ${USER_DIR}/requirements.txt
RUN pip install -r ~/requirements.txt
ENV PYTHONPATH="/home/pyuser/project/src:/home/pyuser/project/test:$PYTHONPATH"
# 不要なら消す
RUN echo 'eval "$(oh-my-posh init bash --config ~/project/.poshthemes/night-owl.omp.json)"' >> ~/.bashrc
RUN echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc


ARG PYTHON_VERSION=3.11.7

ENV PYENV_ROOT="/home/${USERNAME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/home/${USERNAME}/.local/bin:$PATH"

RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && echo "done 1" \
    && pyenv install ${PYTHON_VERSION} \
    && echo "done 2" \
    && pyenv global ${PYTHON_VERSION} \
    && echo "done 3" \
    && pwd
