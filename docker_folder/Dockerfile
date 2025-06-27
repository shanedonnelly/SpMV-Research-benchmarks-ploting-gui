# Utiliser l'image officielle Python slim comme base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier d'abord le fichier des dépendances pour profiter du cache Docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application dans le conteneur
COPY ./app .

# Exposer le port sur lequel Streamlit s'exécute
EXPOSE 8501

# Commande pour lancer l'application
# L'option --server.address=0.0.0.0 est nécessaire pour rendre l'application accessible depuis l'extérieur du conteneur
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]