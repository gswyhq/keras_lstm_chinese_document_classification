FROM tensorflow/tensorflow:1.5.0-py3

#RUN apt-get update && \
#    apt-get -y install \
#    supervisor

WORKDIR /classification

COPY . /classification

RUN cd /classification && pip3 install -r requirements.txt -i https://pypi.douban.com/simple

ENV LANG C.UTF-8

CMD ["/bin/bash"]

# docker build -t classification:20180905 -f Dockerfile .