# CI/CD Base Image
FROM python:3.11.7-slim-bullseye

ARG USERNAME=runner

ENV MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI
ENV MLFLOW_S3_ENDPOINT_URL=MLFLOW_S3_ENDPOINT_URL
ENV AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY

RUN adduser ${USERNAME} && \
    mkdir /home/${USERNAME}/mlbot_public && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    git \
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

# RUN echo 'PATH="$PATH:/home/runner/.local/bin"' >> ~/.bashrc
# RUN source ~/.bashrc

USER ${USERNAME}

WORKDIR /home/${USERNAME}/mlbot_public

COPY ./docker/py_dev/requirements.txt ./requirements.txt

RUN ls && pwd && whoami && pip install -r requirements.txt

ARG PYTHON_VERSION=3.11.7

ENV PYTHONPATH="/home/${USERNAME}/mlbot_public/src:/home/${USERNAME}/mlbot_public/test:$PYTHONPATH"
ENV PYENV_ROOT="/home/${USERNAME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/home/${USERNAME}/.local/bin:$PATH"

RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && echo "done 1" \
    && pyenv install ${PYTHON_VERSION} \
    && echo "done 2" \
    && pyenv global ${PYTHON_VERSION} \
    && echo "done 3"
