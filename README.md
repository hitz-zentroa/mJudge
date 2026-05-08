<p align="center">
    <h2 align="center"> 📊 Towards Reliable Multilingual Judge Models: An Empirical Study </h2>


<p align="center">
    <a href="https://github.com/hitz-zentroa/mJudge/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/mJudge"></a>
    <a href="https://arxiv.org/abs/2406.15227"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
     <br>
</p>

<p align="justify">
This repository contains the code accompanying our paper, "Towards Reliable Multilingual Judge Models: An Empirical Study".

In this work, we study how to build reliable multilingual LLM-as-a-Judge systems when evaluation data is only available in English, with experiments across English, Spanish, and Basque.

We specifically analyze two practical settings:

- **When in-domain training data is available**, where judge models can be fine-tuned directly for the target evaluation task.
- **When in-domain training data is not available**, requiring either zero-shot transfer or fine-tuning on related but out-of-domain data.

For the first scenario, we investigate the effect of instruction translation, monolingual versus multilingual supervision, and model scale. Our experiments show that multilingual training consistently improves performance in Spanish and Basque compared to strictly monolingual training, suggesting beneficial cross-lingual transfer during judge model learning. We further observe that keeping evaluation instructions and rubrics in English often produces more stable multilingual judgments than translating prompts into the target language. Interestingly, smaller open 8B models fine-tuned on in-domain data achieve performance comparable to substantially larger 70B models and rival proprietary judge models, highlighting the effectiveness of efficient open models for multilingual evaluation.

For the latter scenario, we find that larger models in zero-shot settings are generally more robust across domains and languages, while fine-tuning 70B models on mismatched or out-of-domain supervision can significantly degrade evaluation quality. Smaller models benefit from additional supervision on related data, but their overall performance in this setting remains substantially below that of larger zero-shot models.

Overall, our findings provide practical guidance for building efficient and reliable multilingual evaluation pipelines, particularly in scenarios where task-specific multilingual supervision is limited.

---

## Key Components

### `translate/`
Code used to translate train and test datasets.

### `evaluation/`

Code for running inference on benchmark datasets, computing evaluation metrics, and generating the analysis plots used in the paper.

### `llama-recipes/`

Fork and adaptation of Meta’s Llama cookbook used for model fine-tuning on feedback data. This directory retains the original licensing and attribution as provided by Meta. See the corresponding license files for details.

---
## Datasets

English data:

- Train: [Feedback-Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection)
- Test: [RECON](https://huggingface.co/datasets/ai4bharat/recon) and [FLASK](https://github.com/kaistAI/FLASK)

Basque and Spanish Data:

- Train: [es_eu-Feedback-Collection](https://huggingface.co/datasets/HiTZ/es_eu-Feedback-Collection)
- Test: [es_eu-RECON](https://huggingface.co/datasets/HiTZ/es_eu-recon) and [es_eu-FLASK](https://huggingface.co/datasets/HiTZ/es_eu-flask)

Please refer to the original dataset licenses before redistribution or commercial use.

# Citation

```bibtex
@inproceedings{,
  title={Towards Reliable Multilingual Judge Models: An Empirical Study},
  author={},
  year={2026}
}
```
