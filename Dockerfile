FROM python:3.12

WORKDIR /apps

COPY requirements.txt /apps

RUN pip install -r requirements.txt

COPY . /apps

CMD ["python", "kozlova_amo3.py"]
