# Container image for Automa itself (the CLI). See docs/DECISIONS.md D34.
#
# Build:  docker compose build        (or: docker build -t automa:local .)
# Use:    docker compose run --rm automa status
#
# No secrets are baked in: the API key arrives at run time from .env (env_file in
# docker-compose.yml). .dockerignore keeps .env, workspace/ and git data out of the
# build context.
ARG PYTHON_IMAGE=python:3.12-slim

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AUTOMA_WORKSPACE=/data/workspace
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl \
    && rm -rf /wheels \
    && groupadd --gid 1000 automa \
    && useradd --uid 1000 --gid 1000 --home-dir /data --no-create-home \
       --shell /usr/sbin/nologin automa \
    && mkdir -p /data/workspace /data/benchmarks \
    && chown -R 1000:1000 /data
WORKDIR /data
USER 1000:1000
ENTRYPOINT ["automa"]
CMD ["--help"]
