FROM python:3.11.7-slim-bullseye

ARG USERNAME=pyuser
ARG USER_DIR=/home/$USERNAME
ARG WORK_DIR=$USER_DIR/project

USER root

COPY ./entrypoint.sh /tmp/
# COPY ./cron1m.sh /root/script/
# COPY ./cron5m.sh /root/script/
# COPY ./cron15m.sh /root/script/

RUN chmod -R 755 /tmp/entrypoint.sh

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    wget \
    sudo \
    vim \
    cron \
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
    libncursesw5-dev  \
    xz-utils \
    tk-dev  \
    libxml2-dev  \
    libxmlsec1-dev \
    libffi-dev  \
    liblzma-dev

# crontabに登録
# RUN echo '*/1 * * * * sh /root/script/cron1m.sh' > /var/spool/cron/crontabs/root
# RUN echo '*/5 * * * * sh /root/script/cron5m.sh' > /var/spool/cron/crontabs/root
# RUN echo '*/15 * * * * sh /root/script/cron15m.sh' > /var/spool/cron/crontabs/root
# RUN crontab /var/spool/cron/crontabs/root

RUN useradd -m ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    # usermod -s /bin/bash ${USERNAME} \
    mkdir ${WORK_DIR} && \
    chown -R ${USERNAME}:${USERNAME} ${WORK_DIR} && \
    chmod -R 755 ${WORK_DIR}

# USER ${USERNAME}
# RUN mkdir ${USER_DIR}/log && touch ${USER_DIR}/log/recorder.log && pip install -U pip
COPY ./requirements.txt ${USER_DIR}/requirements.txt
RUN mkdir ${USER_DIR}/log && \
    touch ${USER_DIR}/log/recorder.log && \
    pip install -U pip && \
    # pip install -r ${USER_DIR}/requirements.txt -t ${USER_DIR}/.local/lib/python3.11/site-packages && \ # これだとrootではないuserになる
    pip install -r ${USER_DIR}/requirements.txt && \
    chown -R ${USERNAME}:${USERNAME} ${USER_DIR}

ENV PYTHONPATH="/home/pyuser/project/src:/home/pyuser/project/test:$PYTHONPATH"

# USER root
# エントリーポイントの設定
ENTRYPOINT ["/tmp/entrypoint.sh"]

# ADD crontab /var/spool/crontab/${USERNAME}
# RUN crontab /var/spool/crontab/${USERNAME}
