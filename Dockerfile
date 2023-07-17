ARG PORT=443

FROM ubuntu:latest

WORKDIR /app

COPY . .

RUN cd /app && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    poetry install

CMD poetry run python3 .\main.py --host 0.0.0.0 --port $PORT