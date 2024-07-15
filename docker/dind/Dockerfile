FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    vim \
    iputils-ping \
    iptables \
    apt-utils \
    xz-utils \
    dotnet-sdk-6.0 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG DOCKER_CHANNEL=stable
ARG DOCKER_VERSION=25.0.4
ARG DOCKER_COMPOSE_VERSION=2.24.7

ARG IMAGE_NAME
ARG PERSONAL_ACCESS_TOKEN
ARG HOST
ARG ORGANIZATION
ARG REPOSITORY
ENV IMAGE_NAME=${IMAGE_NAME}
ENV PERSONAL_ACCESS_TOKEN=${PERSONAL_ACCESS_TOKEN}
ENV HOST=${HOST}
ENV ORGANIZATION=${ORGANIZATION}
ENV REPOSITORY=${REPOSITORY}

ENV BINARY_URL=https://github.com/actions/runner/releases/download/v2.314.1/actions-runner-linux-x64-2.314.1.tar.gz
ENV RUNNER_NAME=${REPOSITORY}_runner_wsl
ENV RUNNER_GROUP=Default
ENV RUNNER_LABELS="self-hosted,Linux,X64"
ENV RUNNER_WORKDIR=_work
ENV USER=runner

# Install docker
RUN export CHANNEL=${DOCKER_CHANNEL}; \
    export VERSION=${DOCKER_VERSION}; \
    curl -fsSL --retry 3 https://get.docker.com | sh

ENV DOCKER_EXTRA_OPTS "--log-level=error --experimental"

# Install dind hack script
RUN curl -fsSL --retry 3 "https://github.com/moby/moby/raw/v${DOCKER_VERSION}/hack/dind" -o /usr/local/bin/dind \
    && chmod a+x /usr/local/bin/dind

# Install docker-compose
RUN curl -fsSL --retry 3 "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

RUN useradd ${USER} && \
    echo "${USER}:${USER}" | chpasswd && \
    chsh -s /usr/bin/bash ${USER} && \
    usermod -aG sudo ${USER} && \
    mkdir /home/${USER} && \
    mkdir /home/${USER}/actions-runner && \
    chown -R ${USER}:${USER} /home/${USER}

RUN curl -fsSL https://get.docker.com -o /home/${USER}/actions-runner/get-docker.sh && \
    sudo sh /home/${USER}/actions-runner/get-docker.sh && \
    sudo usermod -aG docker ${USER}

RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}/actions-runner

# Install github actions self hosted runner
RUN curl -fsSL -o actions-runner.tar.gz -L $BINARY_URL && \
    tar xf actions-runner.tar.gz && \
    rm actions-runner.tar.gz && \
    echo $PERSONAL_ACCESS_TOKEN && \
    ./config.sh \
    --unattended \
    --url $HOST/$ORGANIZATION/$REPOSITORY \
    --pat $PERSONAL_ACCESS_TOKEN \
    --name $RUNNER_NAME \
    --runnergroup $RUNNER_GROUP \
    --labels $RUNNER_LABELS \
    --work $RUNNER_WORKDIR


VOLUME /var/lib/docker

# ENTRYPOINT [ "/init" ]
