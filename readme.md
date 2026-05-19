项目描述：
# 🧠 医影智元-医学影像处理平台
一个集超分辨率、伪影去除、分割和报告生成功能于一体的智能医疗成像系统。
---

## 📌 引言
该项目是一个全面的医学图像处理工具包，专为 MRI/CT 分析而设计。
它整合了多种深度学习模型，以实现：
- 🔍 图像超分辨率
- 🧹 特征去除
- 🧠 医疗图像分割
- 📄 基于 LLaVA 的 AI 辅助诊断报告生成
该系统是基于 PyQt5 开发的图形用户界面，便于用户进行操作。
---

## 🚀 功能特性
- ✅ 多模态医学图像支持（包括 MRI 和 CT）
- ✅ 基于深度学习的超分辨率重建
- ✅ 用于改善图像质量的伪影去除
- ✅ 基于 Transformer 的分割（例如 TransUNet）
- ✅ 通过本地 LLaVA 模型实现自动报告生成
- ✅ 视觉界面（PyQt5）
---

## ⚙️ Installation
1. Create environment (recommended)
conda create -n medimg python=3.8
conda activate medimg

2. Install dependencies
pip install -r requirements.txt

3. 安装 Ollama
直接下载安装：https://ollama.com/download
ollama pull llava:7b


## ⚙️Run the main program:
python main.py

📁 项目文件说明

本项目整体结构清晰，按功能与模块划分，便于部署、维护与二次开发。各主要文件及目录说明如下：
🧠 1. Source code

该部分为系统核心运行代码，主要包括：

前端界面实现（基于 PyQt5）
功能模块调用逻辑
图像处理与模型推理流程控制
与后端模型及接口的交互实现

👉 该部分为系统的主要执行入口，负责整体业务流程的调度与展示。

🗂️ 2. Dataset sample

用于系统功能测试与演示的数据集合：

包含医学影像相关样本数据
覆盖超分辨、去伪影、分割等任务场景
从完整数据集中随机抽取部分样本构成
用于系统试用、功能验证与效果展示

👉 不包含完整训练集，仅用于快速体验系统功能。

🧩 3. Engineering design documents
系统底层功能实现模块，包括：

各类函数定义文件（function modules）
核心类结构定义（class definitions）
图像预处理与后处理逻辑
模型调用接口封装
工具类与辅助算法模块

👉 该部分构成系统的核心算法与功能实现基础。

🧠 4. Model file

系统各功能对应的深度学习模型文件：

图像超分辨模型（Model.pt）
图像去伪影模型
医学图像分割模型（如 TransUNet / U-Net 等）

👉 所有模型均以 .pt 格式存储，可直接加载推理使用。

📦 5. requirements.txt（依赖环境文件）

项目运行所需的 Python 依赖包清单：

包含 PyTorch、OpenCV、NumPy 等基础库
包含 PyQt5 前端界面依赖
包含图像处理与深度学习相关扩展库
pip install -r requirements.txt
👉 建议使用 Python 3.8–3.10 环境运行以保证兼容性。

---

