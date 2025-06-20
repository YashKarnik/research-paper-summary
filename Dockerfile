FROM python:3.9
WORKDIR /app
RUN mkdir -p data/uploads
RUN mkdir -p data/indexes
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN ls
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD [ "python", "app.py" ]