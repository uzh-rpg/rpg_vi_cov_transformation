This folder contains the implementation of the covariance transformation method in the following publication:

Zichao Zhang, Guillermo Gallego and Davide Scaramuzza, "On the Comparison of Gauge Freedom Handling in Optimization-Based Visual-Inertial State Estimation," in IEEE Robotics and Automation Letters, vol. 3, no. 3, pp. 2710-2717, July 2018.

If the code is used in an academic context, please cite

```latex
@inproceedings{Zhang18ral,
  author={Zichao Zhang and Guillermo Gallego and Davide Scaramuzza},
  journal={IEEE Robotics and Automation Letters},
  title={On the Comparison of Gauge Freedom Handling in Optimization-Based Visual-Inertial State Estimation},
  year={2018},
  month={July},}
```

See http://rpg.ifi.uzh.ch/vi_cov_trans.html for a brief introduction of the covariance transformation.

## Instructions
Simply run

```python
python2 compare_cov_free_fixed.py
```

The transformed covariance/uncertainties will be plotted and saved in `plots`.

The script calls the function in `cov_transformation.py` to transform the free gauge covariance to the gauge fixation case. We provide several example datasets in `data` folder. You can try different ones by changing the code at the beginning of the script `cov_transformation.py`.

The covariance transformation method is implemented in `cov_transformation.py`, and it does not depend on our data and its format. Please see the documentation there for details (e.g., input and output).

