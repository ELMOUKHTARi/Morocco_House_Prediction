# Utiliser une image Python officielle
FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet (templates, static, artifacts, util.py, server.py)
COPY server/ server/

EXPOSE 5000

CMD ["python", "server/server.py"]
