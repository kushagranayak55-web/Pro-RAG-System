# Evaluation

Everything used to measure ProRAG lives here. The app itself is in the repository root.

| File | What it is |
|---|---|
| `prorag_eval.ipynb` | Retrieval and answer-quality benchmark (55 questions, 3 papers). Run in Google Colab. |
| `prorag_extras.ipynb` | Prompt-injection test on the live app and RAGAS faithfulness. Run in Colab. |
| `eval_questions.json` | The 55 questions with gold pages and reference answers. |
| `*.csv` | Raw results from my runs. |

To reproduce: open a notebook in Colab, add your own API keys, and upload the PDFs
(the three papers are on arXiv: 1706.03762, 2005.11401, 2307.09288).
