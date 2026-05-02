FROM ultralytics/ultralytics:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
EXPOSE 8000
CMD ["uvicorn", "src.yolo.infer:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]