FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt requirements-dev.txt ./

# No torch, and no model download step — the ONNX weights are committed to the
# repo and arrive via the COPY below, so builds are offline and reproducible.
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
