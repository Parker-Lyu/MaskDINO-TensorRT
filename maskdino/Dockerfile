FROM nvcr.io/nvidia/pytorch:25.11-py3
WORKDIR /opt/maskdino
ENV MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1
COPY maskdino/requirements.txt maskdino/dependencies.lock.json ./
COPY maskdino/scripts/fetch.py scripts/fetch.py
RUN python scripts/fetch.py sources
RUN python -m pip install --no-build-isolation -r requirements.txt
COPY maskdino/patches patches
RUN git -C third_party/MaskDINO apply /opt/maskdino/patches/maskdino-pytorch.patch \
    && python -m pip install --no-build-isolation --no-deps -e third_party/detectron2 \
    && python -m pip install --no-build-isolation --no-deps -e third_party/panopticapi
RUN cd third_party/MaskDINO/maskdino/modeling/pixel_decoder/ops \
    && python setup.py build install
COPY LICENSE NOTICE maskdino/THIRD_PARTY_NOTICES.md ./
COPY maskdino/licenses licenses
WORKDIR /opt/maskdino/third_party/MaskDINO
CMD ["python", "train_net.py", "--help"]
