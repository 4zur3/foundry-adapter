FROM python:3.11-slim

WORKDIR /app

# (Optionnel) Outils utiles en dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# En dev, on ne copie pas app.py dans l'image (on le montera en volume)
# COPY app.py .

EXPOSE 8000

# --reload = auto-reload quand tu modifies le fichier
# --reload-dir /app = surveille le dossier monté
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]
