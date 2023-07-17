FROM python:3.12.0b4-alpine

WORKDIR /app

# 安装系统依赖
RUN apk update && apk add --no-cache build-base libffi-dev

# 安装并激活虚拟环境
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --upgrade pip

# 使用 poetry 安装项目依赖
COPY pyproject.toml poetry.lock ./
RUN pip install "poetry>=1.3.0,<1.4.0"
RUN poetry config installer.max-workers 10
RUN poetry install -vvv --no-root || poetry install -vvv --no-root || poetry install -vvv --no-root

# 复制其余应用程序文件
COPY . .

EXPOSE 443

CMD ["poetry", "run", "python", "kernel_program/main.py"]
