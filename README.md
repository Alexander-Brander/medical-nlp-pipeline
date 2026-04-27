# Medical NLP Pipeline (MediPublish)

A three-part biomedical NLP coursework project for an MSc Data Science assessment, built around a fictional medical-publishing platform ("MediPublish"). Each notebook tackles a distinct production task on the same corpus of medical abstracts: **named entity recognition**, **document classification**, and **near-duplicate detection**.

The unifying question across the three tasks: *which model families actually pay off on biomedical short text, and where do general-domain or "more powerful" models stop helping?* Findings are mixed - and that is the interesting part.

## Key Results

### 1. Named Entity Recognition
Four NER systems compared on a stratified sample of 2,001 medical abstracts across five conditions, with **400 hand-verified extractions** (100 per model) used for precision.

| Model | Coverage (ents/doc) | Unique Entities | Hand-Verified Precision |
|---|---|---|---|
| spaCy en_core_web_sm (general-domain) | 8.5 | 5,278 | 80.8% (clinically irrelevant) |
| **ScispaCy BC5CDR** | **6.0** | **5,094** | **89.9%** |
| ScispaCy BioNLP13CG | 9.7 | 8,442 | 56.6% |
| d4data biomedical-ner-all (transformer) | 21.3 | 10,855 | 40.4% |

BC5CDR wins the precision-recall trade-off for MediPublish's use case. The transformer extracts the most entities but has a 60% false-positive rate; the general-domain model is precise on its own labels but the labels (DATE, CARDINAL, PERSON) are clinically meaningless.

### 2. Text Classification
Routing medical abstracts to one of five hospital departments. **18 model configurations** evaluated across three representations (TF-IDF, MiniLM sentence embeddings, PubMedBERT) and multiple classifiers (LinearSVC, Logistic Regression, MLP), each on raw and preprocessed text.

| Model | Strict Macro F1 | Weighted F1 | Lenient F1 |
|---|---|---|---|
| MiniLM + LinearSVC | 0.62 | 0.69 | - |
| PubMedBERT + LinearSVC | 0.61 | 0.68 | - |
| **TF-IDF + LinearSVC (preprocessed)** | **0.66** | **0.72** | **0.87** |

5-fold stratified CV on the winner: mean F1 0.78, std 0.004 (very stable). Lenient evaluation (predicting *any* clinically valid department for multi-label abstracts) lifts F1 to 0.87, matching the realistic business requirement. **Sparse term-frequency representations beat dense biomedical transformers on this task** - vocabulary signal dominates over contextual semantics for department routing.

### 3. Duplicate Detection
Cosine-similarity retrieval over 11,227 unique abstracts, evaluated on 1,197 known-duplicate query pairs.

| Method | MRR | MAP | Recall@1 | Recall@5 |
|---|---|---|---|---|
| TF-IDF | 1.0000 | 0.9999 | 0.9307 | 1.0 |
| MiniLM | 1.0000 | 1.0000 | 0.9307 | 1.0 |
| PubMedBERT | 1.0000 | 1.0000 | 0.9307 | 1.0 |
| Fine-tuned PubMedBERT (contrastive) | 1.0000 | 1.0000 | 0.9307 | 1.0 |

All four approaches achieve near-perfect performance. **TF-IDF lexical matching is sufficient** - dense embeddings and contrastive fine-tuning add no measurable improvement on this corpus, suggesting the duplicates are primarily near-exact textual overlap rather than semantic paraphrase. Deployment recommendation: ship the TF-IDF baseline.

## Cross-Cutting Findings

- **"Stronger" model is not always better.** A domain-tuned CNN (BC5CDR) beat a transformer with 84 entity types on precision; sparse TF-IDF beat PubMedBERT on classification; lexical retrieval matched fine-tuned biomedical embeddings on duplicate detection.
- **Hand-verification matters when no ground-truth labels exist.** Surface metrics (entities/doc) painted the transformer as the strongest NER model; manual annotation showed the opposite.
- **Lenient vs. strict evaluation reveals when the metric is wrong, not the model.** A 0.66 strict / 0.87 lenient gap on classification points to multi-label ambiguity in the labels, not classifier weakness.

## Project Structure

```
medical-nlp-pipeline/
├── notebooks/
│   ├── 01_ner_pipeline.ipynb            # spaCy / ScispaCy / transformer NER comparison
│   ├── 02_text_classification.ipynb     # 18-config bake-off for department routing
│   ├── 03_duplicate_detection.ipynb     # TF-IDF vs. SBERT vs. PubMedBERT retrieval
│   ├── verify_general.csv               # Hand-annotated NER samples (general-domain)
│   ├── verify_bc5cdr.csv                # Hand-annotated NER samples (BC5CDR)
│   ├── verify_bionlp.csv                # Hand-annotated NER samples (BioNLP13CG)
│   └── verify_transformer.csv           # Hand-annotated NER samples (transformer)
├── data/                                # Dataset goes here (gitignored - see data/README.md)
├── docs/
│   └── CMM544_NER_Report.pdf            # Written NER report submitted with assessment
├── requirements.txt
├── .gitignore
└── README.md
```

## Data

The corpus is the **Medical Text Classification dataset** (`medical_tc_train.csv`, `medical_tc_test.csv`, `medical_tc_labels.csv`) - 14,438 medical abstracts labelled with one of five conditions:

1. Neoplasms
2. Digestive system diseases
3. Nervous system diseases
4. Cardiovascular diseases
5. General pathological conditions

Mean abstract length: ~150-180 words. Severe class imbalance (label 5 dominates). Source: [TimSchopf/medical_abstracts](https://huggingface.co/datasets/TimSchopf/medical_abstracts) on Hugging Face. Drop the three CSVs into the `data/` folder before running the notebooks (see `data/README.md`).

The four `verify_*.csv` files in `notebooks/` are hand-verified NER samples produced by the NER notebook itself (cell 46 builds the sample, cell 47 reads it back to compute precision after annotation).

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/Alexander-Brander/medical-nlp-pipeline.git
cd medical-nlp-pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# spaCy / scispaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz

# Place medical_tc_*.csv into the data/ folder (see data/README.md)

# Register Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=medical-nlp --display-name="Medical NLP Pipeline"
```

The notebooks were originally run on Google Colab with GPU; cell 6 of the NER notebook and cell 4 of the classification/duplicate notebooks contain a Colab path block plus a commented local-execution block - swap them when running locally.

## Tech Stack

- **Python 3.10+** - core language
- **pandas, numpy** - data manipulation
- **scikit-learn** - TF-IDF, LinearSVC, Logistic Regression, MLP, evaluation, stratified CV
- **NLTK** - tokenisation, stopwords, WordNet lemmatisation
- **spaCy 3.7 / scispaCy 0.5** - rule-based and CNN NER (en_core_web_sm, en_ner_bc5cdr_md, en_ner_bionlp13cg_md)
- **transformers (Hugging Face)** - d4data/biomedical-ner-all token classifier
- **sentence-transformers** - all-MiniLM-L6-v2, S-PubMedBert-MS-MARCO, contrastive fine-tuning via MultipleNegativesRankingLoss
- **torch** - GPU backend for transformer inference and fine-tuning
- **matplotlib, seaborn** - PCA plots, confusion matrices, score distributions

## Limitations

- **No ground-truth NER labels.** Precision is estimated from a 100-sample hand-verification per model, not from a held-out annotated test set.
- **Cross-model agreement (Jaccard) confounds schema disagreement with disagreement on extraction.** A label-mapping step would tighten this.
- **Single-label ground truth on a multi-label problem.** The classification dataset assigns each abstract to exactly one department, but ~41% of test abstracts are clinically valid for multiple departments. The lenient evaluation (F1 0.87) is the more honest performance estimate.
- **No hyperparameter tuning.** Grid/Bayesian search across the TF-IDF n-gram range and SVC C parameter would likely lift the strict F1 a few points further.
- **Duplicate detection ceiling effect.** All four methods score near-perfect; the dataset does not stress-test paraphrase-aware retrieval. A harder benchmark with surface-different but semantically duplicate abstracts would be needed to discriminate the methods.
- **Colab-first paths.** The notebooks assume Google Drive mount; running locally requires un-commenting the alternative path block in the data-loading cells.

## Course Context

Submitted for the **CMM544 Natural Language Processing** module of the MSc Data Science programme at Robert Gordon University, 2026.

