---
license: apache-2.0
widget:
- instruction: How can I stay energized throughout the day?
  response: Drink lots of coffee!
  dataset: open-assistant
language:
- en
datasets:
- OpenAssistant/oasst1
- databricks/databricks-dolly-15k
- HuggingFaceH4/helpful-instructions
tags:
- llm
- human-feedback
- weak supervision
- data filtering
---

## See [our blogpost](https://snorkel.ai/how-we-built-a-better-genai-with-programmatic-data-development/) for a more in-depth discussion of the model.

<br>

# Summary
Instruction tuning has emerged as an important step in developing performant large language models (LLMs) for generative AI tasks. While industry-backed LLMs such as ChatGPT, Bard, Claude, and even the open-source Llama 2 have relied on massive, expensive proprietary datasets unavailable to the public, the open source community has banded together to create similar datasets such as OpenAssistant and Dolly that are available to everyone.  However, high variance in the quality and distribution of responses collected by volunteers has limited the quality of resulting open source models.

This model (1) classifies instructions with a standardized schema that can be applied across datasets and (2) scores response quality on a scale of 0-1. The purpose is to measure and track instruction diversity across training sets, and enable filtering based on response quality for more targeted fine-tuning.

The instruction classification schema is based on prior work in large language models:

* <strong>Open-qa</strong>: question-answering without context, e.g., “When was Google founded?”
* <strong>Closed-qa</strong>: question-answer from a provided context, e.g., “Look at the following paragraph and tell me how many mentions of fruit there are.”
* <strong>Brainstorming</strong>: e.g., “Give me some ideas for planning a beach trip.”
* <strong>Generation</strong>: e.g., “Write me an essay comparing baroque with minimalist music”.
* <strong>Summarization</strong>: e.g., “Summarize the main points from this news article” 
* <strong>Other</strong>: e.g., anything that did not fit the previous five categories.

The response quality model was developed as a binary classlifier ("is/is not an acceptable response"), with the following goals:
  1. Enable filtering of instruction/response datasets for higher quality responses.
  2. Enable this filtering while maintaining (or increasing) instruction diversity.
  3. Develop the model with a lightweight architecture and a scalable data labeling process that requires minimal human-hours.

The response model itself was developed with weak supervision in one day by two FTEs at Snorkel, AI. The model is under continuous development and planned work includes additional datasets and more refined curating labeling functions. We also welcome feedback from the community on any observed patterns of errors!

# Model development
The model pipeline currently consists of a chain of two xbgoost algorithms, one for instruction classification and one for response quality (modeled as binary is/is not an acceptable response classifier). We trained the algorithms with weak supervision techniques and a feature space that includes metadata specific to each dataset, perplexity, measures with [simcse embeddings](https://arxiv.org/pdf/2104.08821.pdf) and attributes involving regex, parts-of-speech tagging and response duration. In order to maintain a lightweight architecture that requires only CPUs for inference, we omitted perplexity from the feature space used to train the xgboost end models (so perplexity was used for weak supervision only).

# Model evaluation
## Instruction classification
Instruction classification scores were measured with ground-truth developed internally, with an out-of-sample accuracy/macro averaged f1 score of 78%/70%. The largest error mode appears linked with basic uncertainty as to how to classify an instruction. For example, "What are a few words that can be used to describe running?" could be interpeted as a ```generation``` task to write a brief snippet describing running, a ```brainstorming``` task to simply come up ideas for writing about running, or (as was indicated in metadata associated with the instruction) as an ```open-qa``` task to answer what running is. However, model predictions appear unbiased when comparing the distributions of ground-truth and predicted classes. Thus, the model remains useful for tracking overall instruction diversity and representation.

## Response quality
Response quality scores were evaluated with double-blind A/B testing that compared dataset responses against what was generated by ChatGPT (version 3.5 turbo). Our evaluation confirmed that response quality predicted preferences for the dataset response over ChatGPT's:

| Model response score      | Win rate over ChatGPT |
| ----------- | ----------- |
| 0-0.25      | 0.25       |
| 0.25-0.5   | 0.28        |
| 0.5-0.75   | 0.43        |
| 0.75-1.0  | 0.47        |

# Usage
The model can accept either a dictionary or list of dicts as input. Each dict needs an ```instruction``` field at a bare minimum (in which case it will simply classify the instruction). If a ```response field``` is included, a response score will be returned. Users can also provide a ```dataset field```, which will only change model predictions if it falls under one of the existing sources we trained on (but can be left blank): dolly, helpful-instructions or open-assistant.

## Example
Input:
<br>
```{'instruction': 'What are ways I can stay energized throughout the day?', 'response': 'Drink lots of coffee!'}```
<br>
<br>
Model output:
<br>
```{'instruction class': 'brainstorming', 'instruction class confidence': 0.9683452, 'response quality': 0.08076164}```