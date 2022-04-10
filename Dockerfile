# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster AS files_image
WORKDIR /mnist
COPY app.py utils.py classifier.h5 requirements.txt ./
COPY static static
COPY templates templates


FROM python:3.8-slim-buster
WORKDIR /mnist-ocr
COPY --from=files_image /mnist .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && pip3 install -r requirements.txt
ENV FLASK_RUN_PORT="80"
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]