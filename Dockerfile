FROM python:3.12.0b4-alpine

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry && poetry install --no-dev

COPY . /app

ENV API_PORT=443

CMD ["poetry", "run", "python", "main.py"]
