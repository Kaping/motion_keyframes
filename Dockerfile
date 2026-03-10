FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_ANALYTICS_ENABLED=False
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-lc"]

# 1. 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    blender git git-lfs ffmpeg wget perl \
    && rm -rf /var/lib/apt/lists/*

# 2. 베이스 이미지에 포함된 PyTorch/torchvision 확인
RUN python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
PY

WORKDIR /root
RUN git lfs install && git clone https://github.com/OpenMotionLab/MotionGPT3.git

WORKDIR /root/MotionGPT3

# 3. Python 의존성 설치 및 환경 설정
RUN pip install --no-cache-dir --upgrade cmake \
    && pip install --no-cache-dir pyarrow --only-binary=:all: \
    && sed -i '/bpy/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r prepare/requirements_render.txt \
    && pip install --no-cache-dir librosa soundfile gdown "gradio==3.50.2" spacy \
    && python -m spacy download en_core_web_sm

# 4. 코드 수정 (Gradio 서버 설정 및 Whisper GPU 사용 설정)
# - 원본 코드의 .to("cpu") 부분을 제거하거나 .to(device)로 유지하여 GPU를 쓰게 합니다.
RUN sed -i 's/server_name="localhost", server_port=8888/server_name="0.0.0.0", server_port=7860/g' app.py && \
    sed -i 's|whisper_path: deps/whisper-large-v2|whisper_path: openai/whisper-tiny|g' configs/assets.yaml

# 5. 실행 명령 (모델 다운로드 스크립트 실행 후 바로 앱 시작)
# 모델이 이미 볼륨에 있으면 스크립트들이 알아서 스킵하거나 덮어씁니다.
ENTRYPOINT ["/bin/bash", "-c", "\
    find prepare/ -name '*.sh' -exec perl -pi -e 's/\\r$//' {} + && \
    bash prepare/download_t2m_evaluators.sh && \
    bash prepare/download_motiongpt_pretrained_models.sh && \
    mkdir -p checkpoints && \
    (ls checkpoints/motiongpt3.ckpt || gdown --id '1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l' -O checkpoints/motiongpt3.ckpt) && \
    bash prepare/download_smpl_model.sh && \
    bash prepare/prepare_gpt2.sh && \
    bash prepare/download_mld_pretrained_models.sh && \
    python -u app.py \
"]

EXPOSE 7860