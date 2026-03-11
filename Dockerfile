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
print("cuda available:", torch.cuda.is_available())
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
    && pip install --no-cache-dir --force-reinstall torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir librosa soundfile gdown "gradio==3.50.2" spacy \
    && python -m spacy download en_core_web_sm

# 4. 코드 수정 (Gradio 서버 설정 및 Whisper 경로만 수정; 디바이스는 원본 로직 유지)
RUN sed -i 's/server_name="localhost", server_port=8888/server_name="0.0.0.0", server_port=7860/g' app.py && \
    sed -i 's|whisper_path: deps/whisper-large-v2|whisper_path: openai/whisper-tiny|g' configs/assets.yaml

# 5. 실행 명령 (모델 다운로드 스크립트 실행 후 바로 앱 시작)
# 재실행 시 이미 내려받은 폴더가 있으면 clone 단계는 스킵합니다.
ENTRYPOINT ["/bin/bash", "-c", "\
    find prepare/ -name '*.sh' -exec perl -pi -e 's/\\r$//' {} + && \
    bash prepare/download_t2m_evaluators.sh && \
    mkdir -p deps/t2m/t2m && \
    mv deps/t2m/t2m/t2m/* deps/t2m/t2m/ 2>/dev/null || true && \
    if [ ! -d checkpoints/MotionGPT-base ]; then bash prepare/download_motiongpt_pretrained_models.sh; else echo '[skip] checkpoints/MotionGPT-base already exists'; fi && \
    mkdir -p checkpoints && \
    (ls checkpoints/motiongpt3.ckpt || gdown --id '1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l' -O checkpoints/motiongpt3.ckpt) && \
    if [ -d deps/smpl_models/smplh ] || [ -d smpl_models/smplh ]; then \
        echo '[skip] SMPL model already exists'; \
    elif [ -f deps/smpl.tar.gz ]; then \
        echo '[local] found deps/smpl.tar.gz, extracting to deps/...'; \
        mkdir -p deps && tar -xzf deps/smpl.tar.gz -C deps; \
    elif [ -f smpl.tar.gz ]; then \
        echo '[local] found smpl.tar.gz, extracting to deps/...'; \
        mkdir -p deps && tar -xzf smpl.tar.gz -C deps; \
    else \
        bash prepare/download_smpl_model.sh; \
    fi && \
    if [ ! -f deps/gpt2/model_state_dict.pth ]; then rm -rf deps/gpt2 deps/mot-gpt2 && bash prepare/prepare_gpt2.sh; else echo '[skip] deps/gpt2/model_state_dict.pth already exists'; fi && \
    bash prepare/download_mld_pretrained_models.sh && \
    rm -rf deps/mot-gpt2 && \
    python -m scripts.gen_mot_gpt && \
    python -u app.py \
"]

EXPOSE 7860
