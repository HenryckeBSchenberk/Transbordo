ARG version=latest

FROM tensorflow/tensorflow:${version} as builder
RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y \
ffmpeg \
libsm6 \
libxext6 

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

FROM tensorflow/tensorflow:${version}

RUN apt-get update

COPY --from=builder /root/.local /root/.local
WORKDIR /app

ENV PATH=/root/.local:$PATH

CMD ["python", "-c", "import cv2; import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"]