services:
  - type: web
    name: foto-verlenger
    env: python
    buildCommand: |
      pip install -r requirements.txt
      apt-get update && apt-get install -y ffmpeg
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
    healthCheckPath: /health
