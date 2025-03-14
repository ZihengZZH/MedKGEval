## Evaluation with MedKGEval (`CPubMedKG_small`)

The experimental results for task-oriented evaluation on `CPubMedKG_small` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| [G] Qwen2-0.5B     | 50.00 | 37.50 | 58.00 | 48.50 | 28.64 | 46.84 | 26.96 | 34.15 | 14.80 | 54.17 | 29.80 | 32.92 | 38.52 |
| [G] Qwen2-1.5B     | 92.39 | 17.50 | 51.00 | 53.63 | 67.66 | 36.83 | 48.54 | 51.01 | 15.29 | 64.31 | 62.83 | 47.48 | 50.71 |
| [G] Qwen2-7B       | 98.91 | 67.50 | 74.00 | 80.14 | 70.91 | 65.23 | 65.85 | 67.33 | 36.92 | 56.72 | 83.35 | 59.00 | 68.82 |
| [G] Baichuan2-7B   | 88.04 | 90.00 | 55.00 | 77.68 | 27.59 | 45.25 | 57.86 | 43.57 | 40.04 | 70.06 | 89.87 | 66.66 | 62.63 |
| [G] Baichuan2-13B  | 96.74 | 70.00 | 67.00 | 77.91 | 43.35 | 66.65 | 56.33 | 55.44 | 39.05 | 69.52 | 71.51 | 60.03 | 64.46 |
| [M] DISC-MedLLM    | 71.74 | 15.00 | 53.00 | 46.58 | 31.51 | 8.09  | 39.88 | 26.49 | 22.84 | 00.21 | 52.16 | 25.07 | 32.71 |
| [M] HuatuoGPT2-7B  | 60.87 | 12.50 | 65.00 | 46.12 | 35.20 | 33.62 | 28.07 | 32.30 | 19.40 | 52.31 | 48.98 | 40.23 | 39.55 |
| [M] HuatuoGPT2-13B | 85.87 | 62.50 | 73.00 | 73.79 | 34.63 | 32.69 | 42.71 | 36.68 | 32.91 | 55.44 | 41.58 | 43.31 | 51.26 |
| [M] PULSE-7B       | 31.52 | 45.00 | 56.00 | 44.17 | 25.75 | 11.69 | 27.37 | 21.60 | 15.80 | 31.30 | 48.63 | 31.91 | 32.56 |
| [M] WiNGPT2        | 97.83 | 85.00 | 73.00 | 85.28 | 75.21 | 31.86 | 57.96 | 55.01 | 33.80 | 68.50 | 73.73 | 58.68 | 66.32 |
| [G] GPT-4o         | 97.83 | 90.00 | 71.00 | 86.28 | 84.12 | 59.31 | 62.19 | 68.54 | 46.51 | 41.50 | 89.29 | 59.10 | 71.31 |
| [G] DeepSeek-V3 ${\color{red}[\text{New}]}$ | 97.83 | 87.50 | 71.00 | 85.44 | 78.35 | 58.48 | 60.95 | 65.93 | 42.82 | 45.72 | 87.10 | 58.55 | 69.97 |

The experimental results for knowledge-oriented evaluation on `CPubMedKG_small` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| [G] Qwen2-0.5B     | 38.36 | 38.34 | 28.65 | 33.11 | 36.60 |
| [G] Qwen2-1.5B     | 51.93 | 51.39 | 46.15 | 47.44 | 50.08 |
| [G] Qwen2-7B       | 69.99 | 68.99 | 58.72 | 61.79 | 66.59 |
| [G] Baichuan2-7B   | 66.08 | 65.20 | 45.28 | 51.28 | 60.56 |
| [G] Baichuan2-13B  | 68.23 | 67.47 | 55.79 | 57.17 | 64.04 |
| [M] DISC-MedLLM    | 32.22 | 32.90 | 20.34 | 22.72 | 29.51 |
| [M] HuatuoGPT2-7B  | 38.74 | 39.01 | 26.28 | 33.02 | 37.01 |
| [M] HuatuoGPT2-13B | 50.66 | 50.28 | 32.22 | 37.08 | 45.88 |
| [M] PULSE-7B       | 31.24 | 29.89 | 25.67 | 24.65 | 28.15 |
| [M] WiNGPT2        | 63.17 | 62.60 | 49.30 | 53.88 | 59.70 |
| [G] GPT-4o         | 69.39 | 67.69 | 61.27 | 60.69 | 65.36 |
| [G] DeepSeek-V3 ${\color{red}[\text{New}]}$ | 68.93 | 67.26 | 59.38 | 59.66 | 64.73 |

## Evaluation with MedKGEval (`CPubMedKG_large`)

The experimental results for task-oriented evaluation on `CPubMedKG_large` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| [G] Qwen2-0.5B     | 53.01 | 31.52 | 51.22 | 45.25 | 29.98 | 47.68 | 27.57 | 35.08 | 13.08 | 57.20 | 29.45 | 33.24 | 37.86 |
| [G] Qwen2-1.5B     | 92.17 | 26.09 | 51.22 | 56.49 | 67.15 | 38.58 | 51.34 | 52.36 | 14.16 | 66.80 | 63.62 | 48.19 | 52.35 |
| [G] Qwen2-7B       | 96.99 | 59.78 | 70.73 | 75.83 | 70.69 | 66.07 | 69.78 | 68.85 | 34.90 | 65.85 | 74.52 | 58.42 | 67.70 |
| [G] Baichuan2-7B   | 83.73 | 69.57 | 53.66 | 68.99 | 34.85 | 44.07 | 63.63 | 47.52 | 39.73 | 66.51 | 81.77 | 62.67 | 59.72 |
| [G] Baichuan2-13B  | 93.37 | 28.26 | 60.67 | 60.77 | 46.01 | 66.86 | 61.64 | 58.17 | 39.14 | 71.48 | 73.65 | 61.42 | 60.12 |
| [M] DISC-MedLLM    | 68.67 | 23.91 | 50.61 | 47.73 | 31.78 | 07.75 | 45.26 | 28.26 | 22.39 | 00.52 | 53.96 | 25.62 | 33.87 |
| [M] HuatuoGPT2-7B  | 54.22 | 06.52 | 64.63 | 41.79 | 38.49 | 33.46 | 29.24 | 33.73 | 18.02 | 51.15 | 37.48 | 35.55 | 37.02 |
| [M] HuatuoGPT2-13B | 85.54 | 46.74 | 68.90 | 67.06 | 34.05 | 33.31 | 47.53 | 38.30 | 31.18 | 62.06 | 44.20 | 45.81 | 50.39 |
| [M] PULSE-7B       | 33.13 | 46.74 | 49.39 | 43.09 | 23.93 | 12.04 | 29.53 | 21.83 | 14.20 | 40.78 | 34.55 | 29.84 | 31.59 |
| [M] WiNGPT2        | 94.58 | 47.83 | 67.99 | 70.13 | 68.03 | 34.28 | 64.70 | 55.67 | 32.66 | 69.13 | 76.47 | 59.42 | 61.74 |

The experimental results for knowledge-oriented evaluation on `CPubMedKG_large` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| [G] Qwen2-0.5B     | 38.48 | 37.86 | 30.71 | 32.96 | 36.23 |
| [G] Qwen2-1.5B     | 52.87 | 52.31 | 46.28 | 49.66 | 51.43 |
| [G] Qwen2-7B       | 69.10 | 68.52 | 61.51 | 63.68 | 66.91 |
| [G] Baichuan2-7B   | 63.74 | 63.21 | 48.99 | 53.35 | 59.93 |
| [G] Baichuan2-13B  | 65.40 | 64.65 | 55.90 | 58.92 | 62.74 |
| [M] DISC-MedLLM    | 33.57 | 33.39 | 23.08 | 25.84 | 30.88 |
| [M] HuatuoGPT2-7B  | 36.76 | 36.11 | 28.37 | 31.73 | 34.65 |
| [M] HuatuoGPT2-13B | 51.88 | 51.22 | 35.83 | 39.40 | 47.28 |
| [M] PULSE-7B       | 31.41 | 29.95 | 25.75 | 24.59 | 28.16 |
| [M] WiNGPT2        | 62.28 | 61.56 | 54.92 | 56.34 | 59.82 |

## Evaluation with MedKGEval (`CMeKG_small`)

The experimental results for task-oriented evaluation on `CMeKG_small` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| [G] Qwen2-0.5B     | 19.38 | - | 66.67 | 43.03 | - | 47.79 | 19.77 | 33.78 | 17.06 | 54.14 | 29.50 | 33.57 | 37.85 |
| [G] Qwen2-1.5B     | 38.44 | - | 50.00 | 44.22 | - | 43.67 | 40.51 | 42.09 | 14.75 | 58.33 | 70.31 | 47.80 | 46.00 |
| [G] Qwen2-7B       | 92.19 | - | 100.0 | 96.10 | - | 70.28 | 61.28 | 65.78 | 40.56 | 67.62 | 82.76 | 63.65 | 70.09 |
| [G] Baichuan2-7B   | 44.06 | - | 50.00 | 47.03 | - | 48.86 | 51.06 | 49.96 | 44.20 | 71.76 | 69.76 | 61.91 | 55.82 |
| [G] Baichuan2-13B  | 76.56 | - | 50.00 | 63.28 | - | 73.93 | 64.90 | 69.42 | 46.66 | 79.87 | 71.80 | 66.11 | 64.83 |
| [M] DISC-MedLLM    | 54.06 | - | 50.00 | 52.03 | - | 3.61  | 39.50 | 21.56 | 22.73 | 00.71 | 53.50 | 25.65 | 28.21 |
| [M] HuatuoGPT2-7B  | 73.44 | - | 66.67 | 70.06 | - | 33.46 | 24.99 | 29.23 | 22.66 | 52.18 | 58.88 | 44.57 | 42.45 |
| [M] HuatuoGPT2-13B | 71.56 | - | 100.0 | 85.78 | - | 35.70 | 44.57 | 40.14 | 38.06 | 63.75 | 50.71 | 50.84 | 54.54 |
| [M] PULSE-7B       | 36.56 | - | 83.33 | 59.95 | - | 10.23 | 32.75 | 21.49 | 18.69 | 61.47 | 59.82 | 46.66 | 42.89 |
| [M] WiNGPT2        | 87.19 | - | 100.0 | 93.60 | - | 37.14 | 68.48 | 52.81 | 43.02 | 77.89 | 91.83 | 70.91 | 68.92 |
| [G] GPT-4o         | 95.00 | - | 100.0 | 97.50 | - | 64.52 | 74.19 | 69.36 | 56.64 | 70.90 | 83.86 | 70.47 | 74.70 |
| [G] DeepSeek-V3 ${\color{red}[\text{New}]}$ | 92.81 | - | 83.33 | 88.07 | - | 64.97 | 74.93 | 69.95 | 51.10 | 69.72 | 80.68 | 67.17 | 71.13 |

The experimental results for knowledge-oriented evaluation on `CMeKG_small` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| [G] Qwen2-0.5B     | 30.01 | 31.20 | 33.16 | 33.92 | 32.10 |
| [G] Qwen2-1.5B     | 44.42 | 45.37 | 45.38 | 45.16 | 45.30 |
| [G] Qwen2-7B       | 68.11 | 69.02 | 54.05 | 65.17 | 67.73 |
| [G] Baichuan2-7B   | 55.11 | 54.82 | 51.54 | 56.82 | 55.49 |
| [G] Baichuan2-13B  | 66.63 | 68.58 | 55.00 | 69.05 | 68.74 |
| [M] DISC-MedLLM    | 27.44 | 29.43 | 18.55 | 23.76 | 27.54 |
| [M] HuatuoGPT2-7B  | 38.29 | 41.64 | 20.78 | 39.80 | 41.03 |
| [M] HuatuoGPT2-13B | 48.96 | 50.97 | 33.41 | 47.87 | 49.94 |
| [M] PULSE-7B       | 33.92 | 36.38 | 25.67 | 36.47 | 36.41 |
| [M] WiNGPT2        | 64.05 | 66.28 | 51.85 | 62.51 | 65.09 |
| [G] GPT-4o         | 72.82 | 73.33 | 58.82 | 68.11 | 71.59 |
| [G] DeepSeek-V3 ${\color{red}[\text{New}]}$ | 73.18 | 72.56 | 61.13 | 67.03 | 70.72 |

## Evaluation with MedKGEval (`CMeKG_large`)

The experimental results for task-oriented evaluation on `CMeKG_large` are presented below.

| LLM | ET | EC | ED | Avg. | RT | FC | RP | Avg. | ER | R1 | R2 | Avg. | Overall |
|-----|---:|---:|---:|-----:|---:|---:|---:|-----:|---:|---:|---:|-----:|--------:|
| [G] Qwen2-0.5B     | 21.60 | 50.00 | 50.00 | 40.53 | 15.38 | 48.01 | 18.77 | 27.39 | 18.12 | 56.73 | 28.13 | 34.33 | 35.11 |
| [G] Qwen2-1.5B     | 40.53 | 00.00 | 50.00 | 30.18 | 19.23 | 43.36 | 33.11 | 31.90 | 14.34 | 61.82 | 71.50 | 49.22 | 37.58 |
| [G] Qwen2-7B       | 96.50 | 16.67 | 80.00 | 64.39 | 46.15 | 71.53 | 59.23 | 58.97 | 46.67 | 83.11 | 81.39 | 70.39 | 61.74 |
| [G] Baichuan2-7B   | 50.00 | 66.67 | 50.00 | 55.56 | 26.92 | 53.36 | 49.21 | 43.16 | 44.78 | 75.52 | 68.57 | 62.96 | 54.72 |
| [G] Baichuan2-13B  | 76.95 | 16.67 | 50.00 | 47.87 | 38.46 | 75.59 | 56.95 | 57.00 | 46.50 | 80.43 | 75.95 | 67.63 | 56.34 |
| [M] DISC-MedLLM    | 57.82 | 16.67 | 50.00 | 41.50 | 26.92 | 02.49 | 32.79 | 20.73 | 23.29 | 01.15 | 52.47 | 25.64 | 25.89 |
| [M] HuatuoGPT2-7B  | 73.05 | 50.00 | 80.00 | 67.68 | 19.23 | 33.82 | 22.14 | 25.06 | 21.95 | 50.81 | 52.10 | 41.62 | 40.82 |
| [M] HuatuoGPT2-13B | 72.02 | 33.33 | 90.00 | 65.12 | 42.31 | 36.59 | 32.91 | 37.27 | 36.21 | 68.02 | 53.93 | 52.72 | 48.95 |
| [M] PULSE-7B       | 32.30 | 00.00 | 100.0 | 44.10 | 07.69 | 09.30 | 26.24 | 14.41 | 17.59 | 76.41 | 59.82 | 51.27 | 36.66 |
| [M] WiNGPT2        | 94.24 | 50.00 | 100.0 | 81.41 | 69.23 | 37.32 | 59.79 | 55.45 | 44.41 | 85.80 | 89.63 | 73.28 | 66.85 |

The experimental results for knowledge-oriented evaluation on `CMeKG_large` are presented below.

| LLM | CovAvg($\mathcal{E}$) | CovDeg($\mathcal{E}$) | CovAvg($\mathcal{R}$) | CovDeg($\mathcal{R}$) | Cov($\mathcal{T}$) |
|-----|----------------------:|----------------------:|----------------------:|----------------------:|-------------------:|
| [G] Qwen2-0.5B     | 30.86 | 32.38 | 31.81 | 34.16 | 32.98 |
| [G] Qwen2-1.5B     | 44.88 | 43.64 | 45.12 | 44.05 | 43.77 |
| [G] Qwen2-7B       | 70.90 | 70.48 | 60.06 | 67.41 | 69.46 |
| [G] Baichuan2-7B   | 57.49 | 56.92 | 56.30 | 56.90 | 56.91 |
| [G] Baichuan2-13B  | 66.50 | 67.02 | 56.51 | 66.44 | 66.83 |
| [M] DISC-MedLLM    | 27.19 | 27.67 | 18.42 | 22.20 | 25.85 |
| [M] HuatuoGPT2-7B  | 37.29 | 38.79 | 21.55 | 37.17 | 38.25 |
| [M] HuatuoGPT2-13B | 47.29 | 47.23 | 34.31 | 45.93 | 46.79 |
| [M] PULSE-7B       | 32.88 | 35.31 | 26.96 | 37.44 | 36.02 |
| [M] WiNGPT2        | 65.43 | 66.35 | 55.56 | 61.20 | 64.64 |
