FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry
RUN poetry install --no-root
COPY . .
CMD ["poetry", "run", "python", "model.py", "train", "--dataset=/data/tsumladvanced/train.csv" , "--dataset=/data/tsumladvanced/test.csv"]