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