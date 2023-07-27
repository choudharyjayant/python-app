FROM python:3.10

WORKDIR /app

COPY ./ ./

RUN /usr/local/bin/python3 -m pip install --upgrade pip &&\
    pip install -r requirements.txt

EXPOSE 5000

CMD ["python3", "main.py"]