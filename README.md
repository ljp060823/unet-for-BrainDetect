***项目用于脑瘤辅助检测与报告生成系统***
**langchain融合改进的UNet分割模型与大语言模型(Qwen3.5_9b + huatuo_QLoRA微调) 并自动生成结构化、客观的辅助参考报告**

*基于传统的unet网络添加注意力模块提高实例分割准确性的同时添加dsconv,在卷积层实现参数的高效减少,采样数据增强 + 混合损失(CrossEntropy + Dice)*
*dsconv参数减少:可看目录下文件use_dsconv.png*
*huatuo_qwen3.5lora模块：/unet-attention-dsconv_github\qwen3.5_9b_huatuo_qlora*
*vllm模型部署: vllm serve /data/qwen3.5_9b_huatuo   --max-model-len 8192 --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --host 0.0.0.0*
demo_url: 

使用 **LangChain** 构建结构化 前缀提示词Prompt Pipeline
调用 **vLLM 量化部署** 的 Qwen-3.5_9b / Huatuo-LoRA 模型
将 UNet 分割结果 + *彩色可视化图像（base64）*喂给多模态模型，实现位置感知的智能报告生成
自动统计病灶面积占比，极大降低幻觉风险
**使用:**
1.用户通过 Gradio 界面上传脑部MRI图像
2.unet-attention-dsconv 实时进行多类病灶分割
3.生成带标注的可视化图像
4.mian.py:LangChain + vLLM 生成结构化辅助报告（包含面积统计、位置描述）
5.gradio_demo.py:前端同时展示原图、可视化结果和报告文本


<!-- 核心视觉模型:Attention+U-Net+DSConv
    大模型:Qwen-3.5_9b / Huatuo医疗数据集QLoRA + vLLM
    框架:LangChain、FastAPI、Gradio
    数据处理:modelscope_huatuo_v1,datsets
    部署:vLLM+fastapi+gradio -->
