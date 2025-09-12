# Acquiescence Bias in Large Language Models

## Abstract
Acquiescence bias, i.e. the tendency of humans to agree with statements in surveys, independent of their actual beliefs, is well researched and documented. Since Large Language Models (LLMs) have been shown to be very influenceable by relatively small changes in input and are trained on human-generated data, it is reasonable to assume that they could show a similar tendency. We present a study investigating the presence of acquiescence bias in LLMs across different models, tasks, and languages (English, German, and Polish). Our results indicate that, contrary to humans, LLMs display a bias towards answering no, regardless of whether it indicates agreement or disagreement.

## Content of the Repository

* [output](output/): Contains the LLM-generated responses as well as the results of their parsing
* [scripts](scripts/): Contains the code that was used to generate the responses (folder inference) and the code that was used for the parsing and evaluation (utils)

## Citation
````
@inproceedings{braun-2025-acquiescence,
  title = {Acquiescence Bias in Large Language Models},
  author = {Braun, Daniel},
  year = {2025},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  url = {https://arxiv.org/abs/2509.08480},
  doi = {10.48550/arXiv.2509.08480}
}
