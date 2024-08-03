# globalign
A library for fast FFT-computed global mutual information-based rigid alignment using the GPU.

Requirements:
Python3.8.8, PyTorch 1.8.1, numpy, torchvision, scipy, sklearn<br>
(CUDA-compatible GPU for GPU acceleration.)

Related to the article (if you use this code, please cite it):<br>
Johan Öfverstedt, Joakim Lindblad, and Nataša Sladoje. Fast computation of mutual information in the frequency domain with applications to global multimodal image alignment. *Pattern Recognition Letters*, Vol. 159, pp. 196-203, 2022. [doi:10.1016/j.patrec.2022.05.022](https://doi.org/10.1016/j.patrec.2022.05.022)<br>

Preprint: https://arxiv.org/abs/2106.14699

To use the library, please see the included example script "example.py".

Main author of the code:<br>
Johan Öfverstedt



## Learn2Reg 2024 - reference sollution for the COMULISSHGBF challenge
```
python -m venv ./venv
. ./venv/bin/activate
pip install -r requirements.txt -r Learn2Reg/requirements.txt

#Download the Dataset for *TASK 3: COMULISglobe SHG-BF*
unzip COMULISSHGBF.zip

#Run globalign/CMIF registration using a rather coarse (fast) search
python Learn2Reg/COMULISSHGBF_2024.py
```

Validation displacement fields are saved to the directory `output`

