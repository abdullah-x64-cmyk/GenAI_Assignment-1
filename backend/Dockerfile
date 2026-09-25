FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

COPY backend/ /workspace/backend/
COPY data/ /workspace/data/
COPY task1_universal/ /workspace/task1_universal/
COPY task2_hard_routing/ /workspace/task2_hard_routing/
COPY task3_soft_moe/ /workspace/task3_soft_moe/
COPY task4_face_sketch/ /workspace/task4_face_sketch/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
