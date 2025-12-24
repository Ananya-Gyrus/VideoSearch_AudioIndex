FROM python:3.8

WORKDIR /main_app


RUN apt update && \
    apt install -y git wget ffmpeg libsm6 libxext6 dmidecode sudo 

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116




COPY cache_dir /main_app/cache_dir
COPY LanguageBind /main_app/LanguageBind

COPY .env .
COPY generate_key.pyc .
COPY app.pyc .
COPY utils.pyc .
COPY languagebind_utils.pyc .


# Create symlink for Faiss
RUN cd /usr/local/lib/python3.8/site-packages/faiss && \
    ln -s swigfaiss.py swigfaiss_avx2.py

WORKDIR /main_app


EXPOSE 5800

ENTRYPOINT ["python", "graphql_app.pyc"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "app.pyc"]

CMD ["--working_dir", "/work_dir", "--batch_size", "64", "--port", "5800"]

