ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.15.0-torch1.9-ubuntu18.04-py37
FROM ${FROM_IMAGE_NAME}

RUN mkdir -p /workspace/llama2_7b
COPY . /workspace/llama2_7b/
COPY pip.conf /root/.config/pip/
WORKDIR /workspace/llama2_7b

ENV VIRTUAL_ENV=/torch/venv3/pytorch
RUN python3 -m venv $VIRTUAL_ENV \
    && rm -rf /workspace/cair_modelzoo \
    && pip install /workspace/llama2_7b/wheel/cndsp-0.8.0-py3-none-any.whl \
    && pip install /workspace/llama2_7b/wheel/transformers-4.28.1-py3-none-any.whl \
    && pip3 install -e .
    
ENV PATH="$VIRTUAL_ENV/bin:$PATH"