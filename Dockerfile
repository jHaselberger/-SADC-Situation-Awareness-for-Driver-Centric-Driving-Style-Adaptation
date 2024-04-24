FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
 
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN apt-get install -y \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*
RUN git config --global --add safe.directory /root/sadc
RUN pip install lightning==2.1.2
RUN pip install wandb==0.16.0
RUN pip install datasets==2.18.0
RUN pip install pandas==2.2.2
 
# For clustering, we use the FAISS package
# To avoid building all from source we use the provided conda packages
# Create the SADC env
RUN conda create --name SADC
 
# Make use of the faster solver
RUN conda install -n SADC -y conda-libmamba-solver==23.7.0
 
# Install all needed packages
RUN conda install -n SADC -y -c conda-forge \
    faiss-gpu==1.7.4 \
    pyyaml==6.0.1\
    loguru==0.7.2 \
    scikit-learn==1.4.1.post1 \
    dask-ml==2023.3.24 \
    scienceplots==2.1.1 \
    matplotlib==3.8.3 \
    tabulate==0.9.0 \
    datasets==2.18.0 \
    --solver=libmamba
