FROM python:3.13-slim
ENV HOME=/tmp PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 STROKESENSE_LOG=/logs/predictions.jsonl
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py ./
COPY .streamlit .streamlit
COPY app app
COPY src src
COPY models models
COPY artifacts artifacts
RUN useradd --create-home --uid 10001 app && mkdir /logs && chown app /logs
USER app
VOLUME /logs
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]
CMD ["python", "run.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
