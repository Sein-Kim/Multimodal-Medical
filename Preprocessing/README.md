# Preprocessing datasets

### How to Run (ADNI dataset)
- Download ADNI dataset from https://adni.loni.usc.edu/.
- Preprocess ADNI dataset (Saving images)
    - Download .nii files from adni site
    - make './ADNI/all_paths.txt' files which "Patient ID" +'\t' + "path of .nii file of patient"
    - then run
<pre><code>
{python adni_atlas.py}
</code></pre>

- After saving images, to get image representation in SimCLR folder run
<pre><code>
{python run_with_pretrain_with_micle.py}
</code></pre>
- After get image representation and non-image feature, then run 
<pre><code>
{python adni_kmeans.py}
</code></pre>

- Then run
<pre><code>
{python main.py}
</code></pre>

### How to Run (OASIS-3 dataset)
- Download OASIS-3 dataset from https://www.oasis-brains.org/.
- Preprocess OASIS-3 dataset (Saving images)
    - Download .nii files from OASIS-3 site
    -make './OASIS/all_paths.txt' file which "Patient ID" + '\t' + "path of .nii file of patient"
    - then run
<pre><code>
{python oasis_atlas.py}
</code></pre>

- After saving images, to get image representation, in SimCLR folder run
<pre><code>
{python run_with_pretrain_with_micle.py}
</code></pre>


- After get image representation and non-image feature, then run
<pre><code>
{python oasis_kmeans.py}
</code></pre>


- Then run
<pre><code>
{python main.py}
</code></pre>

### How to Run (ABIDE dataset)
- Download ABIDE dataset from https://adni.loni.usc.edu/. (Same site with ADNI)
- Preprocess ABIDE dataset (Saving Images)
    - Download .nii files from ABIDE site
    - make './ABIDE/all_paths.txt' files which "Patient ID" + '\t' + "path of .nii file of patient"
    - then run
<pre><code>
{python abide_atlas.py}
</code></pre>

- After saving images, to get image representation, in SimCLR folder run
<pre><code>
{python run_with_pretrain_with_micle.py}
</code></pre>


- After get image representation and non-image feature, then run
<pre><code>
{python abide_kmeans.py}
</code></pre>

- Then run
<pre><code>
{python main.py}
</code></pre>

### How to Run (QIN-Breast dataset)
- Download QIN-Breast dataset from https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast.
- Preprocess QIN-Breast dataset (Savning Images)
    - Download .dcm files from QIN-Breast site
    - make './QIN/all_paths.txt' files which "Patient ID" + '\t' + "path of .dcm file of patient"
    - then run
<pre><code>
{python qin_save.py}
</code></pre>

- After saving images, to get image representation, in SimCLR folder run
<pre><code>
{python run_with_pretrain_with_micle.py}
</code></pre>

- After get image representation and non-image feature, then run
<pre><code>
{python qin_kmeans.py}
</code></pre>

- Then run
<pre><code>
{python main.py}
</code></pre>

### How to Run (CMMD dataset)
- Download CMMD dataset from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508.
- Preprocess CMMD dataset (Saving Images)
    - Download .dcm files form CMMD site
    - make './CMMD/all_paths.txt' files which "Pateint ID" + '\t' + "path of .dcm file of patient"
    - then run
<pre><code>
{python cmmd_save.py}
</code></pre>

- After saving images, to get image representation, in SimCLR folder run
<pre><code>
{python run_with_pretrain_with_micle.py}
</code></pre>

- After get image representation and non-image feature, then run
<pre><code>
{cmmd.ipynb}
</code></pre>

- Then run
<pre><code>
{python main.py}
</code></pre>
