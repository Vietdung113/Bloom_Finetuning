# BLOOM-fine-tuning
Zalo Bloom finetuning

## Installation

```bash
pip install -r requirements.txt
```


## Training

```bash
python finetune_zalo.py \
    --output_dir output
```

## Progress
- [x] Basic train with transformers
- [x] Basic inference
- [ ] Training with [megatron deep](https://github.com/bigscience-workshop/Megatron-DeepSpeed?fbclid=IwAR3K-Jt6pbT_14NeA0p-yYQRhK_LxXLWl7iXq6V-lTLtMQjRi_kf7E091as)
- [ ] Distillation Training for blooom 560