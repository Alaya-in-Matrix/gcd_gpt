# README

Learn greatest common divisor with GPT-like architecture

This work is inspired by the paper [Can transformers learn the greatest common divisor?](https://arxiv.org/pdf/2308.15594v1.pdf), where the training and testing data are sampled with the *stratified* sampling, to make the GCD uniformly distributed in [1,100]

Instead of using a shallow encoder-decoder architecture, I used a deeper GPT-like architecture with 16 layers and 20m+ parameters, it achieves **98.6** accuracy with 3m training points and 10k testing points. 

![](./learning_curve.png)

```bash
bash run.sh
```

