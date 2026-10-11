# data/

Small processed TCGA-LUAD tables produced by `scripts/01_data_download.py` and `scripts/02_preprocessing_qc.py`:

| File | Content |
|---|---|
| `metadata.csv` | sample metadata (576 samples: 517 primary tumours, 59 adjacent normal) |
| `sample_map.csv` | GDC file-to-sample mapping |
| `clinical_data.csv` | GDC clinical records |
| `gene_id_mapping.csv` | Ensembl ID to gene symbol and gene type |

Not included because of size (recreate by running the scripts):

- `raw_counts.csv` and `filtered_counts.csv` (STAR-Counts matrices; steps 01–02)
- `geo/` (GSE31210, GSE68465, GSE72094 downloads; `scripts/09a_geo_validation.py`)

All source data are open-access TCGA and GEO data.
