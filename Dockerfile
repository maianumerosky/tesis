FROM osgeo/gdal:ubuntu-small-latest

RUN apt-get update && apt-get install -y python3-setuptools python3-pip
COPY ./requirements.txt /tesis/requirements.txt
RUN pip3 install -r tesis/requirements.txt && jupyter contrib nbextension install

COPY ./clasificacion_humedales/utils /tesis/clasificacion_humedales/utils
COPY ./clasificacion_humedales/__init__.py /tesis/clasificacion_humedales
COPY ./setup.py /tesis

RUN cd tesis && pip3 install .

WORKDIR /tesis

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]
