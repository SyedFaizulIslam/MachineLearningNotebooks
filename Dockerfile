FROM python:3.8-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./vwModelCleanData.csv /deploy/
COPY ./iptrendmodel.json /deploy/
COPY ./iptrendmodel.h5 /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"] 
CMD ["app.py"]
