FROM python:3.8

RUN mkdir /projet_digit

WORKDIR /projet_digit

ADD ./* /projet_digit/ 

RUN apt-get update

RUN apt-get install 'ffmpeg' 'libsm6' 'libxext6' -y

RUN python -m pip install --upgrade pip

RUN pip3 --no-cache-dir install -r requirements.txt 

CMD ["cnn_mnist.py"]

ENTRYPOINT ["python3"]

