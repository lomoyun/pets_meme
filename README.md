
🐾 Pet Meme Generator Service
📌 项目简介
一个支持 用户上传宠物照片 & 创意模板，通过 AI 自动生成 宠物 Meme 图片 / GIF / 视频 的高并发服务。
基于 FastAPI + Celery + Stable Diffusion 构建，支持分布式扩展和异步任务处理。

🚀 技术架构

模块说明
前端 (React/Vue)
用户上传宠物照片和创意参考图，输入文案并选择输出类型。

FastAPI 网关
提供 REST API / WebSocket 接口，负责任务接收与状态查询。
部署时支持 Nginx + Gunicorn/Uvicorn 负载均衡。

Celery + Redis/RabbitMQ
作为任务队列，支持高并发 & 异步生成。

AI 推理服务

Stable Diffusion + ControlNet → 照片编辑、背景替换、meme 元素生成

Stable Video Diffusion / ModelScope → GIF & 视频生成

DreamBooth / LoRA（可选） → 个性化训练宠物风格

LLM 服务
将用户输入的自然语言描述转化为 AI 模型可理解的 Prompt。

图像 & 视频处理
使用 Pillow、MoviePy、OpenCV 添加字幕、特效与合成。

对象存储 (MinIO/S3)
存储用户上传素材和生成的 Meme 结果，提供下载链接。