FROM python:3.8

RUN mkdir /projet_digit

WORKDIR /projet_digit

ADD * /projet_digit/

COPY model.json /projet_digit/shared/ 

COPY model.h5 /projet_digit/shared/ 

COPY static static

COPY templates templates

RUN apt-get update

RUN apt-get install 'ffmpeg' 'libsm6' 'libxext6' -y

RUN python -m pip install --upgrade pip

RUN pip3 --no-cache-dir install -r requirements.txt 

EXPOSE 5000

CMD ["python3","-m","flask","run","--host=0.0.0.0","--port=5000"]

