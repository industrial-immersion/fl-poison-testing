FROM ubuntu

ENV role=client
ENV poisoned=false

ARG TARGETPLATFORM

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip git curl

COPY requirements.*.txt .

RUN if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        pip3 install -r requirements.amd64.txt --no-cache-dir; \
        else \
        pip3 install -r requirements.arm64.txt --no-cache-dir; \
        fi

COPY . .

CMD python3 runner.py --$role `[ ${poisoned} = True ] && echo --poisoned` --config configs/config.docker.yml
