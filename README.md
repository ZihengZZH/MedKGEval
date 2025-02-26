# MedKGEval

This repository contains the code and benchmarks for the paper _**How Much Medical Knowledge Do LLMs Have? An Evaluation of Medical Knowledge Coverage for LLMs**_, presented at the Proceedings of WWW 2025 (Web4Good Track).

${\color{red}[\text{2025-02-18 New}]}$ We add evaluation results of DeepSeek-V3 [in the following section](#evaluation-results).

## Overview

**MedKGEval** is designed to evaluate the medical knowledge coverage of large language models (LLMs) using established medical knowledge graphs (KGs). This repository provides the necessary tools and datasets to facilitate comprehensive evaluations of LLMs in the medical domain. The overview of **MedKGEval** is shown in the follwing figure.

![](./images/MedKGEval-pipeline.png)

## Medical Knowledge Graphs

In MedKGEval, we utilize **CPubMedKG** and **CMeKG** as baseline knowledge graphs due to their open-source nature and widespread adoption in the Chinese medical field. The source data for these KGs can be found in the following directories:

- **CPubMedKG**: `kg_data/CPubMedKG`
- **CMeKG**: `kg_data/CMeKG`

The downsampling strategy employed for these medical KGs, as detailed in Section 4.1 of the paper, is implemented in the script located at `utils/kg_sample.py`.

You can retrieve the KG data using the `Git LFS` command or download the zipped file via this [Google Drive link](https://drive.google.com/file/d/1nRrhOWlvzdOgpatpQxJP0jiZrnRxhN8v/view?usp=sharing).

## Evaluation Benchmarks

We employ the scripts in `utils/qa_construct.py` to construct various evaluation tasks, including entity-level, relation-level, and subgraph-level tasks, as described in Section 3.2 of the paper.

We have open-sourced the evaluation benchmarks for both **CPubMedKG** and **CMeKG**, which have been downsampled into large and small scales. You can find these benchmarks in the following directories:

- **CPubMedKG** (large/small): `benchmarks/CPubMedKG_large`, `benchmarks/CPubMedKG_small`
- **CMeKG** (large/small): `benchmarks/CMeKG_large`, `benchmarks/CMeKG_small`

You can retrieve the benchmarks using the `Git LFS` command or download the zipped file via this [Google Drive link](https://drive.google.com/file/d/1lRj3BZ31Yad5sWCt1Lk6d1-uwAczWPJZ/view?usp=sharing).

## Evaluating LLMs

### LLM Description and Statistics

The following table summarizes the large language models evaluated in this study, including their parameter counts, domains, and repository/API versions:

| LLM               | #Params | Domain         | Base           | Repository / API Version               |
|-------------------|---------|----------------|----------------|----------------------------------------|
| Qwen2-0.5B        | 0.5B    | General-domain | -              | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |
| Qwen2-1.5B        | 1.5B    | General-domain | -              | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| Qwen2-7B          | 7B      | General-domain | -              | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)     |
| Baichuan2-7B      | 7B      | General-domain | -              | [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)  |
| Baichuan2-13B     | 13B     | General-domain | -              | [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |
| DISC-MedLLM       | 13B     | Medical-domain | Baichuan-13B   | [Flmc/DISC-MedLLM](https://huggingface.co/Flmc/DISC-MedLLM) |
| HuatuoGPT2-7B     | 7B      | Medical-domain | Baichuan2-7B   | [FreedomIntelligence/HuatuoGPT2-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B)   |
| HuatuoGPT2-13B    | 13B     | Medical-domain | Baichuan2-13B  | [FreedomIntelligence/HuatuoGPT2-13B](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-13B) |
| PULSE-7B          | 7B      | Medical-domain | Bloom-7B       | [OpenMEDLab/PULSE-7bv5](https://huggingface.co/OpenMEDLab/PULSE-7bv5) |
| WiNGPT2           | 8B      | Medical-domain | Llama-3-8B     | [winninghealth/WiNGPT2-Llama-3-8B-Chat](https://huggingface.co/winninghealth/WiNGPT2-Llama-3-8B-Chat) |
| GPT-4o            | -       | General-domain | -              | ```2024-10-01-preview``` |
| DeepSeek-V3 ${\color{red}[\text{New}]}$ | 671B    | General-domain | -              | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |

### Running LLMs with MedKGEval Benchmarks

To evaluate the LLMs using the **MedKGEval** benchmarks, you can utilize the scripts located in `scripts/run_*.py`. Below is an example bash script for evaluating `qwen2-7b` on the `CMeKG_small` dataset:

```bash
# List of evaluated LLMs
llm_list=("qwen2-7b")
# List of Medical KG scale
scale_list=("small")
# List of evaluated tasks
data_list=("CMeKG_entity_level_task_ET" "CMeKG_entity_level_task_EC" "CMeKG_entity_level_task_ED" "CMeKG_relation_level_task_RT" "CMeKG_relation_level_task_FC" "CMeKG_relation_level_task_RP" "CMeKG_subgraph_level_task_ER" "CMeKG_subgraph_level_task_R1" "CMeKG_subgraph_level_task_R2")
# Run the evaluation scripts
for llm in ${llm_list[@]}; do
    for scale in ${scale_list[@]}; do
        for data in ${data_list[@]}; do
            CUDA_VISIBLE_DEVICES=0,1 python3 run_qwen.py --model ${llm} --input data/CMeKG_${scale}/${data}.json --output output/CMeKG_${scale}/${data}_${llm}.json
        done
    done
done
```

You can follow the above script to reproduce our experimental results.

## Evaluation Results

We have open-sourced the responses of several LLMs using the MedKGEval framework. For the `CMeKG_small` dataset, the responses from five selected LLMs (Qwen2-0.5B/1.5B/7B and HuatuoGPT2-7B/13B) can be found in the directory `results/CMeKG_small/`.
Both task-oriented and knowledge-oriented evaluation metrics are computed using the script located at `utils/eval.py`.

You can retrieve the benchmarks using the `Git LFS` command or download the zipped file via this [Google Drive link](https://drive.google.com/file/d/1b6lF6DrjDtecXw7niWlcZlUxcgTykC7d/view?usp=sharing).

The experimental results for task-oriented evaluation on `CPubMedKG_small` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| Qwen2-0.5B     | 50.00 | 37.50 | 58.00 | 48.50 | 28.64 | 46.84 | 26.96 | 34.15 | 14.80 | 50.83 | 29.80 | 31.81 | 38.15 |
| Qwen2-1.5B     | 92.39 | 17.50 | 51.00 | 53.63 | 67.66 | 36.83 | 48.54 | 51.01 | 15.29 | 61.04 | 62.83 | 46.39 | 50.34 |
| Qwen2-7B       | 98.91 | 67.50 | 74.00 | 80.14 | 70.91 | 65.23 | 65.85 | 67.33 | 36.92 | 43.22 | 83.35 | 54.50 | 67.32 |
| Baichuan2-7B   | 88.04 | 90.00 | 55.00 | 77.68 | 27.59 | 45.25 | 57.86 | 43.57 | 40.04 | 51.56 | 89.87 | 60.49 | 60.58 |
| Baichuan2-13B  | 96.74 | 70.00 | 67.00 | 77.91 | 43.35 | 66.65 | 56.33 | 55.44 | 39.05 | 49.05 | 71.51 | 53.20 | 62.19 |
| DISC-MedLLM    | 71.74 | 15.00 | 53.00 | 46.58 | 31.51 | 8.09  | 39.88 | 26.49 | 22.84 | 0.51  | 52.16 | 25.17 | 32.75 |
| HuatuoGPT2-7B  | 60.87 | 12.50 | 65.00 | 46.12 | 35.20 | 33.62 | 28.07 | 32.30 | 19.40 | 50.73 | 48.98 | 39.70 | 39.37 |
| HuatuoGPT2-13B | 85.87 | 62.50 | 73.00 | 73.79 | 34.63 | 32.69 | 42.71 | 36.68 | 32.91 | 50.14 | 41.58 | 41.54 | 50.67 |
| PULSE-7B       | 31.52 | 45.00 | 56.00 | 44.17 | 25.75 | 11.69 | 27.37 | 21.60 | 15.80 | 46.35 | 48.63 | 36.93 | 34.23 |
| WiNGPT2        | 97.83 | 85.00 | 73.00 | 85.28 | 75.21 | 31.86 | 57.96 | 55.01 | 33.80 | 49.71 | 73.73 | 52.41 | 64.23 |
| GPT-4o         | 97.83 | 90.00 | 71.00 | 86.28 | 84.12 | 59.31 | 62.19 | 68.54 | 46.51 | 35.56 | 89.29 | 57.12 | 70.65 |
| DeepSeek-V3 ${\color{red}[\text{New}]}$ | 97.83 | 87.50 | 71.00 | 85.44 | 78.35 | 58.48 | 60.95 | 65.93 | 42.82 | 33.14 | 87.10 | 54.35 | 68.57 |

The experimental results for task-oriented evaluation on `CMeKG_small` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| Qwen2-0.5B     | 19.38 | - | 66.67 | 43.03 | - | 47.79 | 19.77 | 33.78 | 17.06 | 50.90 | 29.50 | 32.49 | 37.15 |
| Qwen2-1.5B     | 38.44 | - | 50.00 | 44.22 | - | 43.67 | 40.51 | 42.09 | 14.75 | 51.47 | 70.31 | 45.51 | 44.65 |
| Qwen2-7B       | 92.19 | - | 100.0 | 96.10 | - | 70.28 | 61.28 | 65.78 | 40.56 | 27.73 | 82.76 | 50.35 | 63.39 |
| Baichuan2-7B   | 44.06 | - | 50.00 | 47.03 | - | 48.86 | 51.06 | 49.96 | 44.20 | 50.96 | 69.76 | 54.97 | 51.70 |
| Baichuan2-13B  | 76.56 | - | 50.00 | 63.28 | - | 73.93 | 64.90 | 69.42 | 46.66 | 48.04 | 71.80 | 55.50 | 59.25 |
| DISC-MedLLM    | 54.06 | - | 50.00 | 52.03 | - | 3.61  | 39.50 | 21.56 | 22.73 | 1.10  | 53.50 | 25.78 | 28.31 |
| HuatuoGPT2-7B  | 73.44 | - | 66.67 | 70.06 | - | 33.46 | 24.99 | 29.23 | 22.66 | 50.14 | 58.88 | 43.89 | 42.09 |
| HuatuoGPT2-13B | 71.56 | - | 100.0 | 85.78 | - | 35.70 | 44.57 | 40.14 | 38.06 | 56.26 | 50.71 | 48.34 | 53.10 |
| PULSE-7B       | 36.56 | - | 83.33 | 59.95 | - | 10.23 | 32.75 | 21.49 | 18.69 | 48.23 | 59.82 | 42.45 | 40.34 |
| WiNGPT2        | 87.19 | - | 100.0 | 93.60 | - | 37.14 | 68.48 | 52.81 | 43.02 | 46.94 | 91.83 | 60.60 | 63.30 |
| GPT-4o         | 95.00 | - | 100.0 | 97.50 | - | 64.52 | 74.19 | 69.36 | 56.64 | 43.68 | 83.86 | 61.39 | 69.92 |
| DeepSeek-V3 ${\color{red}[\text{New}]}$ | 92.81 | - | 83.33 | 88.07 | - | 64.97 | 74.93 | 69.95 | 51.10 | 20.38 | 80.68 | 50.72 | 62.81 |

The experimental results for knowledge-oriented evaluation on `CPubMedKG_small` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| Qwen2-0.5B     | 37.82 | 37.94 | 26.88 | 30.52 | 35.46 |
| Qwen2-1.5B     | 50.11 | 49.94 | 42.72 | 45.62 | 48.50 |
| Qwen2-7B       | 66.24 | 65.43 | 51.60 | 54.98 | 61.95 |
| Baichuan2-7B   | 62.51 | 61.53 | 38.01 | 43.59 | 55.55 |
| Baichuan2-13B  | 64.54 | 63.88 | 46.89 | 49.19 | 58.98 |
| DISC-MedLLM    | 32.22 | 32.94 | 20.59 | 22.74 | 29.54 |
| HuatuoGPT2-7B  | 38.99 | 39.30 | 25.15 | 30.29 | 36.30 |
| HuatuoGPT2-13B | 50.02 | 49.92 | 29.51 | 34.01 | 44.62 |
| PULSE-7B       | 31.64 | 31.46 | 25.62 | 25.10 | 29.34 |
| WiNGPT2        | 59.50 | 59.24 | 42.58 | 46.38 | 54.95 |
| GPT-4o         | 66.75 | 65.66 | 55.95 | 55.60 | 62.31 |
| DeepSeek-V3 ${\color{red}[\text{New}]}$ | 65.25 | 64.07 | 51.65 | 53.09 | 60.41 |

The experimental results for knowledge-oriented evaluation on `CMeKG_small` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| Qwen2-0.5B     | 28.34 | 30.18 | 29.38 | 32.81 | 31.05 |
| Qwen2-1.5B     | 41.82 | 43.79 | 43.44 | 43.73 | 43.77 |
| Qwen2-7B       | 61.10 | 61.86 | 47.60 | 57.04 | 60.25 |
| Baichuan2-7B   | 47.09 | 49.54 | 44.90 | 51.92 | 50.33 |
| Baichuan2-13B  | 59.45 | 62.06 | 48.15 | 62.04 | 62.05 |
| DISC-MedLLM    | 27.72 | 29.55 | 19.21 | 23.90 | 27.67 |
| HuatuoGPT2-7B  | 37.86 | 40.77 | 20.61 | 39.05 | 40.19 |
| HuatuoGPT2-13B | 46.37 | 49.02 | 31.57 | 46.31 | 48.12 |
| PULSE-7B       | 29.32 | 33.31 | 23.28 | 33.80 | 33.47 |
| WiNGPT2        | 55.67 | 59.58 | 41.17 | 55.71 | 58.29 |
| GPT-4o         | 66.56 | 67.68 | 54.26 | 62.86 | 66.07 |
| DeepSeek-V3 ${\color{red}[\text{New}]}$ | 64.26 | 63.98 | 52.23 | 57.34 | 61.77 |

## Citation

If you find this repository or the benchmarks useful in your research, please cite our paper as follows:

```
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
