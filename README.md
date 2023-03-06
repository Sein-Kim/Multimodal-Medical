## Heterogeneous Graph Learning for Multi-modal Medical Data Analysis

The official source code for [**Heterogeneous Graph Learning for Multi-modal Medical Data Analysis**](https://arxiv.org/abs/2211.15158) paper, accepted at AAAI 2023.

### Overview
Routine clinical visits of a patient produce not only image data, but also non-image data containing clinical information regarding the patient, i.e., medical data is multi-modal in nature. Such heterogeneous modalities offer different and complementary perspectives on the same patient, resulting in more accurate clinical decisions when they are properly combined. However, despite its significance, how to effectively fuse the multi-modal medical data into a unified framework has received relatively little attention. In this paper, we propose an effective graph-based framework called HetMed (Heterogeneous Graph Learning for Multi-modal Medical Data Analysis) for fusing the multi-modal medical data. Specifically, we construct a multiplex network that incorporates multiple types of non-image features of patients to capture the complex relationship between patients in a systematic way, which leads to more accurate clinical decisions. Extensive experiments on various real-world datasets demonstrate the superiority and practicality of HetMed. 

<img width="400" alt="figure" src="https://user-images.githubusercontent.com/76777494/219549674-4e550a0a-6a5c-4527-a28d-1f88a1939ab2.png">
- Multiple modalities of medical data provide different and complementary views of the same patient.

### Run our framework

- Due to memory and policy problem of datasets, we cannot upload medical image or non-image of datasets.
- In "Preprocessing", we denote the ways to get datasets and how to preprocess the datasets.

- In this circumstance, we upload embeddings of some datasets from Image embedder and non-image information.
- To checkout reproduce our model, use this .plk file run DMGI model.

- ABIDE data sets
<pre><code>
cd MultiplexNetwork
python main.py --data abide --methapath type0,type1,type2,type3 --isSemi --isAttn --sup_coef 1.0
</code></pre>

- CMMD data sets
<pre><code>
cd MultiplexNetwork
python main.py --data cmmd --methapath type0,type1,type2,type3 --isSemi --isAttn --patience 20 --sup_coef 0.01
</code></pre>


### Cite (Bibtex)
- Please refer the following paer, if you fine HetMed useful in your research:
  - Kim, Sein and Lee, Namkyeong and Lee, Junseok and Hyun, Dongmin and Park, Chanyoung. "Heterogeneous Graph Learning for Multi-modal Medical Data Analysis" AAAI 2023.
  - Bibtex
```
@article{kim2022heterogeneous,
  title={Heterogeneous Graph Learning for Multi-modal Medical Data Analysis},
  author={Kim, Sein and Lee, Namkyeong and Lee, Junseok and Hyun, Dongmin and Park, Chanyoung},
  journal={arXiv preprint arXiv:2211.15158},
  year={2022}
}
```
