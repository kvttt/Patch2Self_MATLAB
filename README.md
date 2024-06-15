Unofficial Re-implementation of Patch2Self in MATLAB
====================================================
- Only support OLS and Ridge regression
  - Internally solves the corresponding normal equations using `mldivide`
  - For OLS: `beta = (X.' * X) \ (X.' * y);` i.e., $$\min_\beta (X^TX)^{-1}X^Ty$$
  - For Ridge regression: `beta = (X.' * X + alpha * eye(size(X, 2))) \ (X.' * y);` i.e., $$\min_\beta (X^TX + \alpha I)^{-1}X^Ty$$
- Does not support `clip_negative_vals`, `shift_intensity`, `verbose` (yet)

- Caution: I haven't thoroughly tested the code.
  Feel free to try it out and let me know if the performance is equivalent to that of the official implementation.

References
----------
[Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
                Denoising Diffusion MRI with Self-supervised Learning,
                Advances in Neural Information Processing Systems 33 (2020)

Adopted from
------------
[Patch2Self (DIPY)](https://github.com/dipy/dipy/blob/master/dipy/denoise/patch2self.py)
