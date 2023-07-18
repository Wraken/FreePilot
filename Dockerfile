FROM python:3.10.6

WORKDIR /python-docker

COPY ./requirements.txt requirements.txt

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y software-properties-common && apt-get update

RUN wget -d https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /"
RUN add-apt-repository contrib
RUN apt-get update
RUN apt-get -y install cuda-11-7

RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get install g++-10 gcc-10
RUN CXX=g++-10 CC=gcc-10 LD=g++-10 pip3 install flash-attn==v1.0.3.post0
RUN pip3 install triton==2.0.0.dev20221202 --no-deps

RUN apt remove -y nvidia-*

COPY proxy .

EXPOSE 5000

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "5000", "app:app"]