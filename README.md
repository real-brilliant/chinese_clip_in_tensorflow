# clip_in_tensorflow

- chinese CLIP (Contrastive Language-Image Pre-Training) in tensorflow
<br> - [paper](https://arxiv.org/pdf/2103.00020.pdf)
<br> - [论文速读](https://mp.weixin.qq.com/s/902luOUrKnJ7kXTxviyALQ)
- 官方版本 in Pytorch: https://github.com/openai/CLIP
- 目前仅支持ViT-32 & BERT的版本, 模型下载: 
<br> - [谷歌网盘](https://drive.google.com/drive/folders/17aECcM7m2aflp9Fl6j8yJfZ-ZSn6v07z?usp=sharing)
- Requirements: tensorflow 1.X (>= 1.15)
- 模型说明
<br> - 图片encoder(ViT)为OpenAI的ViT32版本英文预训练结果
<br> - 文字encoder(BERT)基于OpenAI的ViT32版本英文预训练结果, 利用多语言平行语料&知识蒸馏得到, 支持包括中文在内的多种语言, 由huggingface发布。原为torch模型, 本git将图/文encoder统一转换至tf_ckpt, 并集成到同一个checkpoint中
