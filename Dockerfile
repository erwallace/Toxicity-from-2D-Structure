FROM python:3.11

COPY requirements*.in .
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade -r requirements.in
RUN pip install --no-cache-dir --upgrade -r requirements-dev.in
RUN pip install -e .
