FROM python:3.13-slim

WORKDIR /app

# 云端平台通过环境变量注入监听地址；HF Spaces 默认 7860，Render 会注入 PORT
ENV HOST=0.0.0.0
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "webapp.py"]
