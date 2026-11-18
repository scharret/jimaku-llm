FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir requests==2.32.* "fugashi[unidic-lite]==1.4.*" \
 && useradd -u 1000 -m app
WORKDIR /app
COPY subtrans/ /app/subtrans/
USER app
ENTRYPOINT ["python", "-m", "subtrans.main"]
CMD ["scan", "--watch", "900"]
