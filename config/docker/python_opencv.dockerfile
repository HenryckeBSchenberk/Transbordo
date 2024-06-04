ARG version=latest
FROM python:${version} as builder
RUN apt-get update -y && apt-get upgrade -y && \
apt-get install -y \
ffmpeg \
libsm6 \
libxext6 

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

FROM python:${version}

RUN apt-get update

COPY --from=builder /root/.local /root/.local
WORKDIR /app

ENV PATH=/root/.local:$PATH

WORKDIR /app
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && \

RUN apt-get update && apt-get install build-essential  locales-all joe vim -y
RUN apt-get install dialog apt-utils  usbutils -y

ARG USERNAME=dev

ARG PUID=1000

ARG PGID=1000

# RUN groupadd -g ${PGID} ${USERNAME}
# && useradd -u ${PUID} -g ${USERNAME} -d /home/${USERNAME} ${USERNAME} \
# && mkdir /home/${USERNAME} \
# && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

RUN usermod -a -G dialout root && \
usermod -a -G video root && \
# && usermod -a -G dev root \
chown -R root /app
ENV DIR=/home/imcome
RUN echo $DIR

RUN mkdir -p ${DIR} && \
chmod +r ${DIR}

COPY *.deb ${DIR}

# VOLUME [${DIR}]

# WORKDIR ${DIR}
RUN apt-get install ${DIR}/libxcb-*.deb -y
RUN apt-get install ${DIR}/pylon_*.deb -y
RUN apt-get install ${DIR}/codemeter*.deb -y

CMD ["python", "-c", "import cv2; print(cv2.__version__)"]