FROM ubuntu:16.04 

RUN apt-get update && apt-get install -y \
  build-essential \
  vim-gnome \
  git \
  byobu \
  python3 \
  python3-dev \
  python3-pip \
  python3-tk

RUN rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip && pip install -U pip
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
ENV PYTHONIOENCODING=utf-8

ADD . /qprojects
WORKDIR /qprojects




