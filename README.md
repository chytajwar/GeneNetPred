# GeneNetPred

Code accompanying the manuscript "GeneNetPred: A Graph Neural Network for Predicting Cancer Gene Dependency from Multi-Omics Data," prepared for submission to *Computational Biology and Chemistry*.

GeneNetPred is a graph neural network that predicts cancer gene dependency scores by integrating gene expression, somatic mutation, copy number variation, pretrained gene2vec embeddings, and protein-protein interaction structure through a co-dependency-gated GraphSAGE architecture with optional Performer-based attention.

This repository also contains fair, matched-condition reimplementations of two baseline methods, DepGPS and DeepDEP, trained under identical conditions to GeneNetPred for direct comparison.

## What This Repository Contains

- `genenetpred.py`: full training and evaluation pipeline, including GeneNetPred (four architecture variants plus teacher-student distillation), DepGPS (four variants, each with and without cdNS, for eight runs), DeepDEP (two variants), hyperparameter sweeps, ten-seed multi-seed evaluation, an ablation study, and all figure generation.
- `comparison_table.csv`: single-split comparison results across all evaluated architectures, informing Table 2 of the manuscript.
- `requirements.txt`: Python package dependencies.

## Data

This repository does not include raw data files, since they are subject to third-party data use agreements and are too large to host here. All datasets used are publicly available.

| Dataset | Source |
|---|---|
| CRISPR-Cas9 gene dependency scores | DepMap Portal, release 26Q1 |
| Gene expression, somatic mutation, copy number variation | DepMap Portal (CCLE) |
| Protein-protein interaction network | BioGRID, version 5.0.256 |
| Gene2vec embeddings | github.com/jingcheng-du/Gene2Vec |
| Context-dependent nonsynonymous-to-synonymous (cdNS) mutation scores | Supplementary materials of Han et al., *Genome Medicine*, 2024 |
| Cell line metadata | DepMap Model.csv |

After downloading, place the files in a `data/` folder and update the file paths at the top of the script (`DRIVE_FOLDER` and the individual `*_PATH` variables) to match your local setup.

## Installation

```bash
pip install -r requirements.txt
```


A CUDA-capable GPU is required for reasonable training time. The script automatically uses multi-GPU `DataParallel` training when more than one GPU is available.


## Usage

```bash
python genenetpred.py
```

The script is checkpoint-resumable. Hyperparameter sweeps, trained models, and multi-seed results are saved incrementally, so an interrupted run can be resumed safely by re-running the same command.

### Output

- `results_v8/comparison_table_v8.csv`: single-split comparison across all evaluated architectures.
- `figures_v8/`: all manuscript figures, including comparison plots, multi-seed evaluation, ablation study, cancer-type breakdown, attention analysis, and synthetic lethality candidates.
- `checkpoints_v8/`: trained model weights and hyperparameter search results.
- `logs/output_v8.log`: full training log.

## Method Summary

Every architecture (GeneNetPred, DepGPS, DeepDEP) receives an independent hyperparameter search and is evaluated across ten random seeds under matched training conditions, on an identical 80/10/10 train, validation, and test split (852 cancer cell lines, 17,401 genes). Results are reported as the mean and standard deviation of test-set Pearson correlation.

## Citation

If you use this code, please cite the associated manuscript. Citation details will be added upon publication.

## License

MIT License. See the `LICENSE` file.
