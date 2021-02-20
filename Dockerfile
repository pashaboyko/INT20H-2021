FROM python:3.8-slim
MAINTAINER Boiko Pavlo 'pboyko172839465@gmail.com'

WORKDIR /usr/src/app
COPY ./INT20H-2021/requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt 
COPY ./INT20H-2021 .

ENTRYPOINT  [ "python","-m", "app"]

