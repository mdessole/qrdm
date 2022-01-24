# QR factorization with Deviation Maximization pivoting

QR factorization with Deviation Maximization pivoting C implementation and Python wrapper. 
Implementation used in

> Monica Dessole, Fabio Marcuzzi "**[Deviation Maximization for Rank-Revealing QR Factorizations](http://arxiv.org/abs/2106.03138)**", Preprint, 2021.

Compile Python module with 

```console
python build_QRDM.py
```

Please, specify:
- setup_QRDM.py, line 14: installation path for BLAS, LAPACK and LAPACKE
- QRDM_wrapper.c, line 1: header file "Python.h"

## Dependencies

- Python 3
- LAPACK
- LAPACKE
- CBLAS
