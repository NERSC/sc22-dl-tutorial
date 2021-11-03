FROM nvcr.io/nvidia/pytorch:21.10-py3

# Upgrade pip
RUN python -m pip install -U setuptools pip

# Install pip dependencies
RUN pip install ruamel.yaml && \
    pip install h5py

# Install benchy lib
RUN git clone https://github.com/romerojosh/benchy.git && \
    cd benchy && \
    python setup.py install && \
    cd ../ && rm -rf benchy
