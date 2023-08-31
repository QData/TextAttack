# Reactive Adversarial Defense

This folder shows the example of using the reactive adversarial defense method to defend against adversarial attacks.

## Introduction
Recent studies have shown that large pre-trained language models are vulnerable to adversarial attacks. Existing methods attempt to reconstruct the adversarial examples. However, these methods usually have limited performance in defense against adversarial examples, while also negatively impacting the performance on natural examples. To overcome this problem, we propose a method called Reactive Perturbation Defocusing (RPD). RPD uses an adversarial detector to identify adversarial examples and reduce false defenses on natural examples. Instead of reconstructing the adversaries, RPD injects safe perturbations into adversarial examples to distract the objective models from the malicious perturbations. Our experiments on three datasets, two objective models, and various adversarial attacks show that our proposed framework successfully repairs up to approximately 97% of correctly identified adversarial examples with only about a 2% performance decrease on natural examples. We also provide a demo of adversarial detection and repair based on our work.

## Requirements
- Python 3.9
- PyTorch 
- transformers
- pyabsa >= 2.0.6

## Usage
### 1. Install pyabsa and textattack
```bash
pip install transformers
pip install pyabsa
pip install textattack
```

### 2. Run the adversarial attack against the defense of Rapid
This will attack the model `tadbert-sst2` with the `textfooler` recipe on the `sst2` dataset. The attack will be defended by the defense of Rapid.
The available models and datasets can be found in [Textattack](https://github.com/yangheng95/TextAttack/blob/9eeef7950d07b7470e92c9e918a7097086eee062/textattack/model_args.py#L96) 
You can report the adversarial attack performance under the defense of Rapid by running the following command:
```bash
textattack attack --recipe textfooler --model tadbert-sst2 --num-examples 100 --dataset sst2 --attack-n 1 
```

## Script examples
Please find the script examples in [examples](https://github.com/yangheng95/TextAttack/blob/master/examples/reactive_defense/sst2_reactive_defense.py)

### 4. Play with the Demo
You can play with the demo on Huggingface space. The [demo](https://huggingface.co/spaces/anonymous8/Rapid-Textual-Adversarial-Defense) is based on the [PyABSA](https://github.com/yangheng95/PyABSA) abd [Textattack](https://github.com/QData/TextAttack).


## Citation
If you find this repo helpful, please cite the following paper:
```
@article{DBLP:journals/corr/abs-2305-04067,
  author       = {Heng Yang and
                  Ke Li},
  title        = {Reactive Perturbation Defocusing for Textual Adversarial Defense},
  journal      = {CoRR},
  volume       = {abs/2305.04067},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.04067},
  doi          = {10.48550/arXiv.2305.04067},
  eprinttype    = {arXiv},
  eprint       = {2305.04067},
  timestamp    = {Thu, 11 May 2023 15:54:24 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-04067.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Author
This work is presented by [Heng Yang](https://github.com/yangheng95). If you have any question, please contact hy345@exeter.ac.uk