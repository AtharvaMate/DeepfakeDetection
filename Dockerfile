FROM python:3.13

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD streamlit run app.py