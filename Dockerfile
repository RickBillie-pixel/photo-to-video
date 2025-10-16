FROM python:3.11-slim

# Installeer FFmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopieer requirements en installeer dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer applicatie code
COPY main.py .

# Expose poort (Render gebruikt de PORT environment variable)
EXPOSE 8000

# Start commando - gebruik $PORT van Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
