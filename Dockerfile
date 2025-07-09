FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ECOAI_PASSWORD=change_me
EXPOSE 8000
CMD ["uvicorn", "your_module:app", "--host", "0.0.0.0", "--port", "8000"]
