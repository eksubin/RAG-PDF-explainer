FROM python:3.9

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y sqlite3 libsqlite3-dev \
    && mkdir /db \
    && /usr/bin/sqlite3 /db/app.db

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["chainlit", "run"]

CMD ["app.py"]