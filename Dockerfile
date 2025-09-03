# 1. 베이스 이미지 (PyTorch 2.3.0 + CUDA 12.1 + cuDNN 8)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# 2. 환경 변수 설정
ENV NVM_DIR="/home/appuser/.nvm"
ENV PATH="/opt/poetry/bin:/home/appuser/.local/bin:${NVM_DIR}/versions/node/v20.16.0/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
ENV ZENML_ANALYTICS_OPT_IN=false
ENV ZENML_DEBUG=true
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true



# 3. 이후 RUN 명령을 bash로 실행하기 위해 쉘 변경
SHELL ["/bin/bash", "-c"]

# 4. root 권한으로 필수 시스템 패키지 설치 및 Rust/Poetry 설치
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git ca-certificates build-essential cmake pkg-config \
      libffi-dev libssl-dev libxml2-dev libxslt1-dev zlib1g-dev \
      libmagic1 libmagic-dev \
  && rm -rf /var/lib/apt/lists/*

# Rust (일부 토치/토크나이저 빌드 대비) + Poetry
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y \
  && echo 'source "$HOME/.cargo/env"' >> /etc/bash.bashrc \
  && curl -sSL https://install.python-poetry.org | python3 -

# 5) 사용자/작업 디렉토리
RUN useradd --create-home --shell /bin/bash appuser \
  && mkdir -p /app \
  && chown -R appuser:appuser /app

# 6. 작업 디렉토리 지정
WORKDIR /app

# 7. non-root 사용자로 전환
USER appuser

# 8. nvm(Node Version Manager), Node.js, Gemini CLI 설치
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    . "$NVM_DIR/nvm.sh" && \
    nvm install 20 && \
    nvm use 20 && \
    npm install -g @google/gemini-cli

# 9. 의존성 파일 복사 (빌드 캐시 활용)
COPY --chown=appuser:appuser pyproject.toml poetry.lock* ./

# 10. Poetry를 사용하여 Python 의존성 설치 (dev 그룹 제외, 가상환경 생성)
RUN poetry --version \
  && poetry config virtualenvs.create true \
  && poetry config virtualenvs.in-project true \
  && poetry install --no-root --only main --sync --no-interaction

ENV VIRTUAL_ENV=/app/.VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:/home/appuser/.local/bin:/opt/poetry/bin:${NVM_DIR}/versions/node/v20.16.0/bin:${PATH}"
# 12. 나머지 애플리케이션 코드 복사
COPY --chown=appuser:appuser . /app

# 13. 기본쉘
CMD ["bash"]
