# Implementation of EasyControl

EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer

<a href='https://arxiv.org/pdf/2503.07027'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

> *[Yuxuan Zhang](https://xiaojiu-z.github.io/YuxuanZhang.github.io/), [Yirui Yuan](https://github.com/Reynoldyy), [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), [Haofan Wang](https://haofanwang.github.io/), [Jiaming Liu](https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en)*
> <br>
> Tiamat AI, ShanghaiTech University, National University of Singapore, Liblib AI

<img src='assets/teaser.jpg'>

## Abstract
Recent advancements in Unet-based diffusion models, such as ControlNet and IP-Adapter, have introduced effective spatial and subject control mechanisms. However, the DiT (Diffusion Transformer) architecture still struggles with efficient and flexible control. To tackle this issue, we propose EasyControl, a novel framework designed to unify condition-guided diffusion transformers with high efficiency and flexibility. Our framework is built on three key innovations. First, we introduce a lightweight Condition Injection LoRA Module. This module processes conditional signals in isolation, acting as a plug-and-play solution. It avoids modifying the base model’s weights, ensuring compatibility with customized models and enabling the flexible injection of diverse conditions. Notably, this module also supports harmonious and robust zero-shot multi-condition generalization, even when trained only on single-condition data. Second, we propose a Position-Aware Training Paradigm. This approach standardizes input conditions to fixed resolutions, allowing the generation of images with arbitrary aspect ratios and flexible resolutions. At the same time, it optimizes computational efficiency, making the framework more practical for real-world applications. Third, we develop a Causal Attention Mechanism combined with the KV Cache technique, adapted for conditional generation tasks. This innovation significantly reduces the latency of image synthesis, improving the overall efficiency of the framework. Through extensive experiments, we demonstrate that EasyControl achieves exceptional performance across various application scenarios. These innovations collectively make our framework highly efficient, flexible, and suitable for a wide range of tasks.

<img src='assets/method.jpg'>

## Todo List
1. - [x] Inference code 
2. - [ ] Pre-trained weights 
3. - [ ] Training code


## Citation
```
@misc{zhang2025easycontroladdingefficientflexible,
      title={EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer}, 
      author={Yuxuan Zhang and Yirui Yuan and Yiren Song and Haofan Wang and Jiaming Liu},
      year={2025},
      eprint={2503.07027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07027}, 
}
```
