FROM python:3

WORKDIR /usr/src/app
COPY *.py ./
COPY ./templates ./templates
COPY ./input ./input
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libspatialindex-dev \
    wkhtmltopdf
CMD ["python", "./slice.py"]
