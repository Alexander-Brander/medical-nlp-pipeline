# Data

The notebooks expect three CSVs in this folder:

```
data/
├── medical_tc_train.csv
├── medical_tc_test.csv
└── medical_tc_labels.csv
```

The CSVs are gitignored (the dataset is not redistributable through this repo) - download them from Hugging Face and drop them here before running the notebooks:

**Source:** https://huggingface.co/datasets/TimSchopf/medical_abstracts

## Schema

- `medical_tc_train.csv` - ~11,550 rows. Columns: `condition_label` (1-5), `medical_abstract` (text).
- `medical_tc_test.csv` - ~2,888 rows. Same schema.
- `medical_tc_labels.csv` - 5 rows mapping `condition_label` to `condition_name` (Neoplasms, Digestive system diseases, Nervous system diseases, Cardiovascular diseases, General pathological conditions).

## Path configuration

Each notebook has a Colab path block at the top (mounting Google Drive) and a commented local-execution block. To run locally, comment the Colab block and uncomment the local block - it points at `data/medical_tc_*.csv` relative to the repo root.
