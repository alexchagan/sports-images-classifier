FROM tensorflow/tensorflow:latest

WORKDIR /usr/src/app

COPY . .

RUN apt-get update

RUN pip install -r /usr/src/app/requirements.txt

ENTRYPOINT [ "python3", "/usr/src/app/trainer.py" ]