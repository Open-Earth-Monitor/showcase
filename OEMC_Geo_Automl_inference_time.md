
# GeoAML - AutoML pipeline evaluation based on inference time

The Open-Earth-Monitor Cyberinfrastructure project has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No. 101059548.

One of the goals of the OEMC project is streamlining ML procudures against geospatial data. Geospatial application of ML often has the end goal of mapping predictions across a large spatial (or spatiotemporal) extent immediately upon fitting a model. This is often necessary even in the prototyping phase to assess problems with the modelling approach which might not be apparent from the training dataset. In these situations the area that needs to be mapped often contains order of magnitude more samples than the training dataset, and the time needed to produce (infer) a full map can be a significant hurdle to the usfulness of models.

To overcome this in an AutoML pipeline, we can directly include inference time as a metric to be tracked and optimized for. We demonstrate this with two prominent AutoML frameworks: [PyCaret](pycaret.gitbook.io/) and [FLAML](https://microsoft.github.io/FLAML/).

## Setting up

This notebook requires the following libraries installed in your environment:
  - `scikit-learn`
  - `xgboost`
  - `lightgbom`
  - `numpy`
  - `pandas`
  - `requests`
  - `pycaret`
  - `flaml[automl]`

You can either `pip install` or `conda install` these yourself, or if you are working from a constarined environment like Colab, uncomment the install lines in the following code block.

Additionally we provide a small module called `automl-utils` included in the notebook repository. This module contains some thin wrappers and helper to assist with integrating an inference time metric with AutoML (most notably when working with PyCaret, which doesn't normally pass the estimator to metric functions). You can install this package from git (by uncommenting the corresponding line in the following code cell), or from a local copy of the notebook repository (by running `python -m pip install .` from the `automl-utils` directory).


```python
### uncomment the following to install dependencies
!python -m pip install scikit-learn xgboost lightgbm numpy pandas requests pycaret flaml[automl]

### uncomment the following to install the automl-utils module
!python -m pip install git+https://github.com/Open-Earth-Monitor/showcase#egg=automl-utils&subdirectory=automl-utils
```

First, we need to fetch some data to work with. The dataset presented here was prepared within the context of [OEMC Hackthon 2023](https://earthmonitor.org/events/hackathon2023/). We can download it from Zenodo and inspect it with Pandas.


```python
import pandas as pd
import requests

ZENODO_URL = "https://zenodo.org/records/13874505"

DATA_FILE = "./train.csv"

resp = requests.get(f"{ZENODO_URL}/files/train.csv?download=1")

with open(DATA_FILE, "wb") as dst:
    dst.write(resp.content)

df_train = pd.read_csv(DATA_FILE)

df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample_id</th>
      <th>station</th>
      <th>month</th>
      <th>fapar</th>
      <th>modis_blue</th>
      <th>modis_red</th>
      <th>modis_nir</th>
      <th>modis_mir</th>
      <th>modis_evi</th>
      <th>modis_ndvi</th>
      <th>...</th>
      <th>dtm_slope</th>
      <th>dtm_aspect-cosine</th>
      <th>dtm_aspect-sine</th>
      <th>dtm_downlslope.curvature</th>
      <th>dtm_upslope.curvature</th>
      <th>dtm_elevation</th>
      <th>dtm_cti</th>
      <th>dtm_neg.openess</th>
      <th>dtm_pos.openess</th>
      <th>dtm_vbf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>52</td>
      <td>2</td>
      <td>0.310634</td>
      <td>235.0</td>
      <td>545.0</td>
      <td>1306.0</td>
      <td>1414.0</td>
      <td>1484.0</td>
      <td>4108.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>-4753.0</td>
      <td>-876.0</td>
      <td>-13.0</td>
      <td>13.0</td>
      <td>351.0</td>
      <td>-2414.0</td>
      <td>153.0</td>
      <td>155.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>14</td>
      <td>9</td>
      <td>0.699500</td>
      <td>355.0</td>
      <td>531.0</td>
      <td>3348.0</td>
      <td>786.0</td>
      <td>5060.0</td>
      <td>7156.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>-3071.0</td>
      <td>945.0</td>
      <td>-22.0</td>
      <td>40.0</td>
      <td>125.0</td>
      <td>-412.0</td>
      <td>152.0</td>
      <td>155.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>52</td>
      <td>3</td>
      <td>0.353572</td>
      <td>276.0</td>
      <td>642.0</td>
      <td>1496.0</td>
      <td>1364.0</td>
      <td>1614.0</td>
      <td>4086.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>-4753.0</td>
      <td>-876.0</td>
      <td>-13.0</td>
      <td>13.0</td>
      <td>351.0</td>
      <td>-2414.0</td>
      <td>153.0</td>
      <td>155.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>73</td>
      <td>11</td>
      <td>0.260067</td>
      <td>519.0</td>
      <td>1196.0</td>
      <td>3256.0</td>
      <td>1247.0</td>
      <td>3112.0</td>
      <td>4628.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>5685.0</td>
      <td>788.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>584.0</td>
      <td>3795.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>257.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>14</td>
      <td>3</td>
      <td>0.779333</td>
      <td>327.0</td>
      <td>528.0</td>
      <td>3106.0</td>
      <td>988.0</td>
      <td>4572.0</td>
      <td>7048.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>-3071.0</td>
      <td>945.0</td>
      <td>-22.0</td>
      <td>40.0</td>
      <td>125.0</td>
      <td>-412.0</td>
      <td>152.0</td>
      <td>155.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3456</th>
      <td>3456</td>
      <td>7</td>
      <td>10</td>
      <td>0.027000</td>
      <td>811.0</td>
      <td>1169.0</td>
      <td>1800.0</td>
      <td>2672.0</td>
      <td>1163.0</td>
      <td>2045.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>-9449.0</td>
      <td>-38.0</td>
      <td>-1.0</td>
      <td>3.0</td>
      <td>1652.0</td>
      <td>100.0</td>
      <td>156.0</td>
      <td>157.0</td>
      <td>373.0</td>
    </tr>
    <tr>
      <th>3457</th>
      <td>3457</td>
      <td>26</td>
      <td>8</td>
      <td>0.036196</td>
      <td>563.0</td>
      <td>1366.0</td>
      <td>2776.0</td>
      <td>2432.0</td>
      <td>1999.0</td>
      <td>3316.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4335.0</td>
      <td>900.0</td>
      <td>-6.0</td>
      <td>-1.0</td>
      <td>1322.0</td>
      <td>2011.0</td>
      <td>157.0</td>
      <td>156.0</td>
      <td>540.0</td>
    </tr>
    <tr>
      <th>3458</th>
      <td>3458</td>
      <td>56</td>
      <td>8</td>
      <td>0.969277</td>
      <td>167.0</td>
      <td>250.0</td>
      <td>3366.0</td>
      <td>525.0</td>
      <td>5704.0</td>
      <td>8624.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>-3085.0</td>
      <td>416.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2786.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>653.0</td>
    </tr>
    <tr>
      <th>3459</th>
      <td>3459</td>
      <td>23</td>
      <td>3</td>
      <td>0.536160</td>
      <td>257.0</td>
      <td>542.0</td>
      <td>2104.0</td>
      <td>887.0</td>
      <td>2918.0</td>
      <td>6168.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>9548.0</td>
      <td>-270.0</td>
      <td>-8.0</td>
      <td>3.0</td>
      <td>42.0</td>
      <td>400.0</td>
      <td>157.0</td>
      <td>156.0</td>
      <td>875.0</td>
    </tr>
    <tr>
      <th>3460</th>
      <td>3460</td>
      <td>73</td>
      <td>1</td>
      <td>0.205879</td>
      <td>801.0</td>
      <td>1528.0</td>
      <td>2659.0</td>
      <td>1740.0</td>
      <td>1781.0</td>
      <td>2701.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>5685.0</td>
      <td>788.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>584.0</td>
      <td>3795.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>257.0</td>
    </tr>
  </tbody>
</table>
<p>3461 rows × 36 columns</p>
</div>



We'll separate some columns from the dataset for later and delete the ones we don't currently need to simplify things. After that we can begin building our AutoML pipeline, first with PyCaret.


```python
groups = df_train.station.copy()
X = df_train[df_train.columns[4:]]

del df_train["sample_id"], df_train["month"], df_train["station"]

df_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fapar</th>
      <th>modis_blue</th>
      <th>modis_red</th>
      <th>modis_nir</th>
      <th>modis_mir</th>
      <th>modis_evi</th>
      <th>modis_ndvi</th>
      <th>modis_lst_day_p05</th>
      <th>modis_lst_day_p50</th>
      <th>modis_lst_day_p95</th>
      <th>...</th>
      <th>dtm_slope</th>
      <th>dtm_aspect-cosine</th>
      <th>dtm_aspect-sine</th>
      <th>dtm_downlslope.curvature</th>
      <th>dtm_upslope.curvature</th>
      <th>dtm_elevation</th>
      <th>dtm_cti</th>
      <th>dtm_neg.openess</th>
      <th>dtm_pos.openess</th>
      <th>dtm_vbf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.310634</td>
      <td>235.0</td>
      <td>545.0</td>
      <td>1306.0</td>
      <td>1414.0</td>
      <td>1484.0</td>
      <td>4108.0</td>
      <td>13656.0</td>
      <td>14032.0</td>
      <td>14384.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>-4753.0</td>
      <td>-876.0</td>
      <td>-13.0</td>
      <td>13.0</td>
      <td>351.0</td>
      <td>-2414.0</td>
      <td>153.0</td>
      <td>155.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.699500</td>
      <td>355.0</td>
      <td>531.0</td>
      <td>3348.0</td>
      <td>786.0</td>
      <td>5060.0</td>
      <td>7156.0</td>
      <td>14904.0</td>
      <td>15200.0</td>
      <td>15296.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>-3071.0</td>
      <td>945.0</td>
      <td>-22.0</td>
      <td>40.0</td>
      <td>125.0</td>
      <td>-412.0</td>
      <td>152.0</td>
      <td>155.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.353572</td>
      <td>276.0</td>
      <td>642.0</td>
      <td>1496.0</td>
      <td>1364.0</td>
      <td>1614.0</td>
      <td>4086.0</td>
      <td>14400.0</td>
      <td>14480.0</td>
      <td>14888.0</td>
      <td>...</td>
      <td>10.0</td>
      <td>-4753.0</td>
      <td>-876.0</td>
      <td>-13.0</td>
      <td>13.0</td>
      <td>351.0</td>
      <td>-2414.0</td>
      <td>153.0</td>
      <td>155.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.260067</td>
      <td>519.0</td>
      <td>1196.0</td>
      <td>3256.0</td>
      <td>1247.0</td>
      <td>3112.0</td>
      <td>4628.0</td>
      <td>13264.0</td>
      <td>13696.0</td>
      <td>13976.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>5685.0</td>
      <td>788.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>584.0</td>
      <td>3795.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>257.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.779333</td>
      <td>327.0</td>
      <td>528.0</td>
      <td>3106.0</td>
      <td>988.0</td>
      <td>4572.0</td>
      <td>7048.0</td>
      <td>15136.0</td>
      <td>15200.0</td>
      <td>15296.0</td>
      <td>...</td>
      <td>11.0</td>
      <td>-3071.0</td>
      <td>945.0</td>
      <td>-22.0</td>
      <td>40.0</td>
      <td>125.0</td>
      <td>-412.0</td>
      <td>152.0</td>
      <td>155.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3456</th>
      <td>0.027000</td>
      <td>811.0</td>
      <td>1169.0</td>
      <td>1800.0</td>
      <td>2672.0</td>
      <td>1163.0</td>
      <td>2045.0</td>
      <td>14456.0</td>
      <td>14832.0</td>
      <td>15272.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>-9449.0</td>
      <td>-38.0</td>
      <td>-1.0</td>
      <td>3.0</td>
      <td>1652.0</td>
      <td>100.0</td>
      <td>156.0</td>
      <td>157.0</td>
      <td>373.0</td>
    </tr>
    <tr>
      <th>3457</th>
      <td>0.036196</td>
      <td>563.0</td>
      <td>1366.0</td>
      <td>2776.0</td>
      <td>2432.0</td>
      <td>1999.0</td>
      <td>3316.0</td>
      <td>15472.0</td>
      <td>15792.0</td>
      <td>16248.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>4335.0</td>
      <td>900.0</td>
      <td>-6.0</td>
      <td>-1.0</td>
      <td>1322.0</td>
      <td>2011.0</td>
      <td>157.0</td>
      <td>156.0</td>
      <td>540.0</td>
    </tr>
    <tr>
      <th>3458</th>
      <td>0.969277</td>
      <td>167.0</td>
      <td>250.0</td>
      <td>3366.0</td>
      <td>525.0</td>
      <td>5704.0</td>
      <td>8624.0</td>
      <td>14680.0</td>
      <td>14960.0</td>
      <td>15056.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>-3085.0</td>
      <td>416.0</td>
      <td>-4.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>2786.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>653.0</td>
    </tr>
    <tr>
      <th>3459</th>
      <td>0.536160</td>
      <td>257.0</td>
      <td>542.0</td>
      <td>2104.0</td>
      <td>887.0</td>
      <td>2918.0</td>
      <td>6168.0</td>
      <td>14504.0</td>
      <td>14832.0</td>
      <td>14952.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>9548.0</td>
      <td>-270.0</td>
      <td>-8.0</td>
      <td>3.0</td>
      <td>42.0</td>
      <td>400.0</td>
      <td>157.0</td>
      <td>156.0</td>
      <td>875.0</td>
    </tr>
    <tr>
      <th>3460</th>
      <td>0.205879</td>
      <td>801.0</td>
      <td>1528.0</td>
      <td>2659.0</td>
      <td>1740.0</td>
      <td>1781.0</td>
      <td>2701.0</td>
      <td>12656.0</td>
      <td>13152.0</td>
      <td>13232.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>5685.0</td>
      <td>788.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>584.0</td>
      <td>3795.0</td>
      <td>157.0</td>
      <td>157.0</td>
      <td>257.0</td>
    </tr>
  </tbody>
</table>
<p>3461 rows × 33 columns</p>
</div>



## PyCaret

PyCaret is a highly accessible AutoML framework designed for rapid experimentation and protyping. It handles many of the common preprocessing tasks internally (to a degree), like data cleaning and feature engineering, and makes it simple to build a workable prototype model relatively quickly, even on lower-end hardware.

We will be using PyCaret's OOP API which allows us to contain a full AutoML experiment in a single object. We will also inspect what kind of metrics the experiment tracks by default.


```python
from pycaret.regression import RegressionExperiment

exp = RegressionExperiment()

# setting up an experiment also outputs a summary of the setup
exp.setup(
    df_train,
    target="fapar",
    fold_groups=groups,
    fold_strategy="groupkfold",
)

# inspect the metrics included by default
exp.get_metrics()
```


<style type="text/css">
#T_5d3f2_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_5d3f2">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5d3f2_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_5d3f2_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5d3f2_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_5d3f2_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_5d3f2_row0_col1" class="data row0 col1" >7872</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_5d3f2_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_5d3f2_row1_col1" class="data row1 col1" >fapar</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_5d3f2_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_5d3f2_row2_col1" class="data row2 col1" >Regression</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_5d3f2_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_5d3f2_row3_col1" class="data row3 col1" >(3461, 33)</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_5d3f2_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_5d3f2_row4_col1" class="data row4 col1" >(3461, 33)</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_5d3f2_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_5d3f2_row5_col1" class="data row5 col1" >(2422, 33)</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_5d3f2_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_5d3f2_row6_col1" class="data row6 col1" >(1039, 33)</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_5d3f2_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_5d3f2_row7_col1" class="data row7 col1" >32</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_5d3f2_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_5d3f2_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_5d3f2_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_5d3f2_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_5d3f2_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_5d3f2_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_5d3f2_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_5d3f2_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_5d3f2_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_5d3f2_row12_col1" class="data row12 col1" >GroupKFold</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_5d3f2_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_5d3f2_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_5d3f2_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_5d3f2_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_5d3f2_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_5d3f2_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_5d3f2_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_5d3f2_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_5d3f2_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_5d3f2_row17_col1" class="data row17 col1" >reg-default-name</td>
    </tr>
    <tr>
      <th id="T_5d3f2_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_5d3f2_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_5d3f2_row18_col1" class="data row18 col1" >eb09</td>
    </tr>
  </tbody>
</table>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Display Name</th>
      <th>Score Function</th>
      <th>Scorer</th>
      <th>Target</th>
      <th>Args</th>
      <th>Greater is Better</th>
      <th>Custom</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mae</th>
      <td>MAE</td>
      <td>MAE</td>
      <td>&lt;function mean_absolute_error at 0x7f50ade44f40&gt;</td>
      <td>neg_mean_absolute_error</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>MSE</td>
      <td>MSE</td>
      <td>&lt;function mean_squared_error at 0x7f50ade45300&gt;</td>
      <td>neg_mean_squared_error</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>RMSE</td>
      <td>RMSE</td>
      <td>&lt;function mean_squared_error at 0x7f50ade45300&gt;</td>
      <td>neg_root_mean_squared_error</td>
      <td>pred</td>
      <td>{'squared': False}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>r2</th>
      <td>R2</td>
      <td>R2</td>
      <td>&lt;function r2_score at 0x7f50ade45b20&gt;</td>
      <td>r2</td>
      <td>pred</td>
      <td>{}</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rmsle</th>
      <td>RMSLE</td>
      <td>RMSLE</td>
      <td>&lt;function RMSLEMetricContainer.__init__.&lt;local...</td>
      <td>make_scorer(root_mean_squared_log_error, great...</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mape</th>
      <td>MAPE</td>
      <td>MAPE</td>
      <td>&lt;function MAPEMetricContainer.__init__.&lt;locals...</td>
      <td>make_scorer(mean_absolute_percentage_error, gr...</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



PyCaret allows us to easily add custom metrics to the experiment, but does not natively pass the estimator object to the metric function (which is necessary to measure inference time). We will circumvent this by using the `automl_utils` module, and add an instance of its `InferenceTimer` as a custom metric to the experiment.


```python
import automl_utils

inference_time_metric_pycaret = automl_utils.InferenceTimer(
    X,  # specify the dataset for timed inference
    target_lib="pycaret",  # specify that the metric will be used with PyCaret
)

exp.add_metric(
    "inference_time",
    "inference time",
    inference_time_metric_pycaret,
    greater_is_better=False,
)

exp.get_metrics()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Display Name</th>
      <th>Score Function</th>
      <th>Scorer</th>
      <th>Target</th>
      <th>Args</th>
      <th>Greater is Better</th>
      <th>Custom</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mae</th>
      <td>MAE</td>
      <td>MAE</td>
      <td>&lt;function mean_absolute_error at 0x7f50ade44f40&gt;</td>
      <td>neg_mean_absolute_error</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mse</th>
      <td>MSE</td>
      <td>MSE</td>
      <td>&lt;function mean_squared_error at 0x7f50ade45300&gt;</td>
      <td>neg_mean_squared_error</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rmse</th>
      <td>RMSE</td>
      <td>RMSE</td>
      <td>&lt;function mean_squared_error at 0x7f50ade45300&gt;</td>
      <td>neg_root_mean_squared_error</td>
      <td>pred</td>
      <td>{'squared': False}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>r2</th>
      <td>R2</td>
      <td>R2</td>
      <td>&lt;function r2_score at 0x7f50ade45b20&gt;</td>
      <td>r2</td>
      <td>pred</td>
      <td>{}</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>rmsle</th>
      <td>RMSLE</td>
      <td>RMSLE</td>
      <td>&lt;function RMSLEMetricContainer.__init__.&lt;local...</td>
      <td>make_scorer(root_mean_squared_log_error, great...</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mape</th>
      <td>MAPE</td>
      <td>MAPE</td>
      <td>&lt;function MAPEMetricContainer.__init__.&lt;locals...</td>
      <td>make_scorer(mean_absolute_percentage_error, gr...</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>inference_time</th>
      <td>inference time</td>
      <td>inference time</td>
      <td>&lt;automl_utils.InferenceTimer object at 0x7f50a...</td>
      <td>make_scorer(inference_time, greater_is_better=...</td>
      <td>pred</td>
      <td>{}</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



To work with this setup, we also need to patch the estimator objects we will use for the experiment using the `automl_utils` module. This will work with any `scikit-learn` compatible model classes.


```python
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

estimators = [
    automl_utils.patch_estimator(e)
    for e in (
        LGBMRegressor(),
        XGBRegressor(),
        RandomForestRegressor(),
    )
]

estimators
```




    [LGBMRegressor_Patched4InferenceTimer(),
     XGBRegressor_Patched4InferenceTimer(base_score=None, booster=None,
                                         callbacks=None, colsample_bylevel=None,
                                         colsample_bynode=None,
                                         colsample_bytree=None, device=None,
                                         early_stopping_rounds=None,
                                         enable_categorical=False, eval_metric=None,
                                         feature_types=None, gamma=None,
                                         grow_policy=None, importance_type=None,
                                         interaction_constraints=None,
                                         learning_rate=None, max_bin=None,
                                         max_cat_threshold=None,
                                         max_cat_to_onehot=None, max_delta_step=None,
                                         max_depth=None, max_leaves=None,
                                         min_child_weight=None, missing=nan,
                                         monotone_constraints=None,
                                         multi_strategy=None, n_estimators=None,
                                         n_jobs=None, num_parallel_tree=None,
                                         random_state=None, ...),
     RandomForestRegressor_Patched4InferenceTimer()]



We can now run our AutoML experiment and get a ranking of our models based on multiple metrics, including inference time.


```python
best = exp.compare_models(
    include=estimators,
    turbo=True,
)

best
```






<style type="text/css">
#T_4459a th {
  text-align: left;
}
#T_4459a_row0_col0, #T_4459a_row0_col6, #T_4459a_row0_col7, #T_4459a_row1_col0, #T_4459a_row1_col1, #T_4459a_row1_col2, #T_4459a_row1_col3, #T_4459a_row1_col4, #T_4459a_row1_col5, #T_4459a_row1_col7, #T_4459a_row2_col0, #T_4459a_row2_col1, #T_4459a_row2_col2, #T_4459a_row2_col3, #T_4459a_row2_col4, #T_4459a_row2_col5, #T_4459a_row2_col6 {
  text-align: left;
}
#T_4459a_row0_col1, #T_4459a_row0_col2, #T_4459a_row0_col3, #T_4459a_row0_col4, #T_4459a_row0_col5, #T_4459a_row1_col6, #T_4459a_row2_col7 {
  text-align: left;
  background-color: yellow;
}
#T_4459a_row0_col8, #T_4459a_row2_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_4459a_row1_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_4459a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4459a_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_4459a_level0_col1" class="col_heading level0 col1" >MAE</th>
      <th id="T_4459a_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_4459a_level0_col3" class="col_heading level0 col3" >RMSE</th>
      <th id="T_4459a_level0_col4" class="col_heading level0 col4" >R2</th>
      <th id="T_4459a_level0_col5" class="col_heading level0 col5" >RMSLE</th>
      <th id="T_4459a_level0_col6" class="col_heading level0 col6" >MAPE</th>
      <th id="T_4459a_level0_col7" class="col_heading level0 col7" >inference time</th>
      <th id="T_4459a_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4459a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_4459a_row0_col0" class="data row0 col0" >Light Gradient Boosting Machine</td>
      <td id="T_4459a_row0_col1" class="data row0 col1" >0.0707</td>
      <td id="T_4459a_row0_col2" class="data row0 col2" >0.0158</td>
      <td id="T_4459a_row0_col3" class="data row0 col3" >0.1074</td>
      <td id="T_4459a_row0_col4" class="data row0 col4" >0.7490</td>
      <td id="T_4459a_row0_col5" class="data row0 col5" >0.0746</td>
      <td id="T_4459a_row0_col6" class="data row0 col6" >0.5389</td>
      <td id="T_4459a_row0_col7" class="data row0 col7" >0.0255</td>
      <td id="T_4459a_row0_col8" class="data row0 col8" >15.4080</td>
    </tr>
    <tr>
      <th id="T_4459a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_4459a_row1_col0" class="data row1 col0" >Extreme Gradient Boosting</td>
      <td id="T_4459a_row1_col1" class="data row1 col1" >0.0740</td>
      <td id="T_4459a_row1_col2" class="data row1 col2" >0.0181</td>
      <td id="T_4459a_row1_col3" class="data row1 col3" >0.1141</td>
      <td id="T_4459a_row1_col4" class="data row1 col4" >0.7200</td>
      <td id="T_4459a_row1_col5" class="data row1 col5" >0.0785</td>
      <td id="T_4459a_row1_col6" class="data row1 col6" >0.4245</td>
      <td id="T_4459a_row1_col7" class="data row1 col7" >0.0200</td>
      <td id="T_4459a_row1_col8" class="data row1 col8" >0.4770</td>
    </tr>
    <tr>
      <th id="T_4459a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_4459a_row2_col0" class="data row2 col0" >Random Forest Regressor</td>
      <td id="T_4459a_row2_col1" class="data row2 col1" >0.0731</td>
      <td id="T_4459a_row2_col2" class="data row2 col2" >0.0180</td>
      <td id="T_4459a_row2_col3" class="data row2 col3" >0.1111</td>
      <td id="T_4459a_row2_col4" class="data row2 col4" >0.7094</td>
      <td id="T_4459a_row2_col5" class="data row2 col5" >0.0766</td>
      <td id="T_4459a_row2_col6" class="data row2 col6" >0.4457</td>
      <td id="T_4459a_row2_col7" class="data row2 col7" >0.1036</td>
      <td id="T_4459a_row2_col8" class="data row2 col8" >0.8630</td>
    </tr>
  </tbody>
</table>

## FLAML

FLAML is modular and efficient AutoML framework with support for a broad range of ML tasks. It is one of the few frameworks in this space that passes the estimator to metric functions, allowing for direct inference time optimization without any helpers. However, we will again use the `automl_utils.InferenceTimer` helper for convenience.

Let's set up our AutoML experiment with FLAML.


```python
from flaml import AutoML

inference_time_metric_flaml = automl_utils.InferenceTimer(
    X,
    target_lib="flaml",  # specify that the metric will be used with FLAML
)

settings = {
    "time_budget": 10,
    "metric": inference_time_metric_flaml,  # define metric as optimization target
    "estimator_list": [  # specify estimator types
        "lgbm",
        "rf",
        "xgboost",
    ],
    "task": "regression",
}

automl = AutoML(**settings)

automl
```


We can now use the `automl` object to optimize an ML pipeline purely for inference time.


```python
y = df_train["fapar"]

automl.fit(X, y)
```

    [flaml.automl.logger: 06-03 21:49:39] {1752} INFO - task = regression
    [flaml.automl.logger: 06-03 21:49:39] {1763} INFO - Evaluation method: holdout
    [flaml.automl.logger: 06-03 21:49:39] {1862} INFO - Minimizing error metric: customized metric
    [flaml.automl.logger: 06-03 21:49:39] {1979} INFO - List of ML learners in AutoML Run: ['lgbm', 'rf', 'xgboost']
    [flaml.automl.logger: 06-03 21:49:39] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:39] {2417} INFO - Estimated sufficient time budget=438s. Estimated necessary time budget=0s.
    [flaml.automl.logger: 06-03 21:49:39] {2466} INFO -  at 0.1s,	estimator lgbm's best error=0.0028,	best estimator lgbm's best error=0.0028
    [flaml.automl.logger: 06-03 21:49:39] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.0020,	best estimator lgbm's best error=0.0020
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.3s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.3s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 6, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.6s,	estimator xgboost's best error=0.0022,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.7s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 8, current learner rf
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.8s,	estimator rf's best error=0.0161,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 9, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.8s,	estimator lgbm's best error=0.0019,	best estimator lgbm's best error=0.0019
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 10, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 0.9s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 11, current learner rf
    [flaml.automl.logger: 06-03 21:49:40] {2466} INFO -  at 1.0s,	estimator rf's best error=0.0161,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:40] {2282} INFO - iteration 12, current learner rf
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.2s,	estimator rf's best error=0.0161,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.3s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 14, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.4s,	estimator xgboost's best error=0.0022,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.4s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 16, current learner rf
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.5s,	estimator rf's best error=0.0152,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.6s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 18, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.6s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 19, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.6s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 20, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.6s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.7s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.7s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 23, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.7s,	estimator lgbm's best error=0.0018,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 24, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.8s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 25, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.8s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 26, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.9s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 27, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 1.9s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 28, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 2.0s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0018
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 29, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 2.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 30, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 2.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 31, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:41] {2466} INFO -  at 2.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:41] {2282} INFO - iteration 32, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 33, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 34, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 35, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.3s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 36, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 37, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 38, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 39, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 40, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.5s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 42, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 43, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.6s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 44, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 45, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 46, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.7s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 47, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 48, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 49, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.8s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 50, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 51, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 52, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 53, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 54, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 2.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 55, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 3.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 56, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 3.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 57, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:42] {2466} INFO -  at 3.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:42] {2282} INFO - iteration 58, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 59, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 60, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 61, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 62, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.3s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 63, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 64, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 65, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.4s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 66, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 67, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 68, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 69, current learner rf
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.7s,	estimator rf's best error=0.0152,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 70, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 71, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.8s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 72, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 73, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.9s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 74, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 75, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 3.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 76, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 4.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 77, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 4.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 78, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 4.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 79, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 4.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 80, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:43] {2466} INFO -  at 4.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:43] {2282} INFO - iteration 81, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 82, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 83, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 84, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 85, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 86, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 87, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 88, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 89, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 90, current learner rf
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.6s,	estimator rf's best error=0.0152,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 91, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 92, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 93, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 94, current learner rf
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.8s,	estimator rf's best error=0.0152,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 95, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 96, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 97, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 4.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 98, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 5.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 99, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 5.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 100, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 5.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 101, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 5.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 102, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:44] {2466} INFO -  at 5.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:44] {2282} INFO - iteration 103, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 104, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 105, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 106, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 107, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 108, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 109, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 110, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 111, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 112, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.5s,	estimator xgboost's best error=0.0020,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 113, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 114, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 115, current learner rf
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.7s,	estimator rf's best error=0.0152,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 116, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 117, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 118, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 119, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 120, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 121, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 122, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 123, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 124, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 5.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 125, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 6.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 126, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 6.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 127, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 6.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 128, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 6.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 129, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:45] {2466} INFO -  at 6.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:45] {2282} INFO - iteration 130, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 131, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 132, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 133, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 134, current learner rf
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.4s,	estimator rf's best error=0.0150,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 135, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 136, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 137, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 138, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 139, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 140, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 141, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 142, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 143, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 144, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 145, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 146, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 147, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 148, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 149, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 150, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 151, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 152, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 6.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 153, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 7.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 154, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 7.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 155, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 7.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 156, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 7.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 157, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:46] {2466} INFO -  at 7.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:46] {2282} INFO - iteration 158, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 159, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 160, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 161, current learner rf
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.4s,	estimator rf's best error=0.0150,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 162, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 163, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 164, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 165, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 166, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 167, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 168, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 169, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 170, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 171, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 172, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 173, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 174, current learner rf
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.9s,	estimator rf's best error=0.0150,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 175, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 7.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 176, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 177, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 178, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 179, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 180, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 181, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:47] {2466} INFO -  at 8.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:47] {2282} INFO - iteration 182, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 183, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 184, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 185, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 186, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 187, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 188, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 189, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 190, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 191, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 192, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 193, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 194, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 195, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 196, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 197, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 198, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 199, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 200, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 201, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 202, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 203, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 204, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 205, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 206, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 207, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 208, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 8.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 209, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 9.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 210, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 9.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 211, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 9.0s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 212, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 9.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 213, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:48] {2466} INFO -  at 9.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:48] {2282} INFO - iteration 214, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 215, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 216, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 217, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.2s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 218, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 219, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.3s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 220, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 221, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 222, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 223, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 224, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 225, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 226, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.5s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 227, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 228, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 229, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 230, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.6s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 231, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 232, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 233, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.7s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 234, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 235, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 236, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 237, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.8s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 238, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 9.9s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:49] {2282} INFO - iteration 239, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:49] {2466} INFO -  at 10.1s,	estimator lgbm's best error=0.0017,	best estimator lgbm's best error=0.0017
    [flaml.automl.logger: 06-03 21:49:50] {2724} INFO - retrain lgbm for 0.1s
    [flaml.automl.logger: 06-03 21:49:50] {2727} INFO - retrained model: LGBMRegressor(colsample_bytree=0.9229314154488576, learning_rate=1.0,
                  max_bin=127, min_child_samples=19, n_estimators=4, n_jobs=-1,
                  num_leaves=5, reg_alpha=0.0009765625,
                  reg_lambda=7.133067957061659, verbose=-1)
    [flaml.automl.logger: 06-03 21:49:50] {2009} INFO - fit succeeded
    [flaml.automl.logger: 06-03 21:49:50] {2010} INFO - Time taken to find the best model: 9.64775013923645


While this works, optimizing an ML pipeline purely for inference time obviously isn't particularly useful. Instead, we can define a more complex metric that combines a traditional loss function with inference time.


```python
import numpy as np

# if we omit the target_lib argument, we get a generic inference timer
# that takes only an estimator and outputs number of seconds elapsed
inference_timer = automl_utils.InferenceTimer(X)


def custom_metric_flaml(X_val, y_val, estimator, *args, **kwargs):
    y_pred = estimator.predict(X_val)
    mae = np.abs(y_pred - y_val).mean()
    seconds_elapsed = inference_timer(estimator)

    # combine MAE and inference time into a single metric to optimize for
    loss = mae * seconds_elapsed * 1000  # magnify by 1000 for more readable results

    # return a single loss value as the optimization target
    # and a dict of metrics to display during optimization
    return loss, {
        "MAE": mae,
        "inference_time": seconds_elapsed,
    }


automl.fit(X, y, metric=custom_metric_flaml)
```

    [flaml.automl.logger: 06-03 21:49:50] {1752} INFO - task = regression
    [flaml.automl.logger: 06-03 21:49:50] {1763} INFO - Evaluation method: holdout
    [flaml.automl.logger: 06-03 21:49:50] {1862} INFO - Minimizing error metric: customized metric
    [flaml.automl.logger: 06-03 21:49:50] {1979} INFO - List of ML learners in AutoML Run: ['lgbm', 'rf', 'xgboost']
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 0, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2417} INFO - Estimated sufficient time budget=651s. Estimated necessary time budget=1s.
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.8768,	best estimator lgbm's best error=0.8768
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 1, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.8768,	best estimator lgbm's best error=0.8768
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 2, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.2s,	estimator lgbm's best error=0.4448,	best estimator lgbm's best error=0.4448
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 3, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.3s,	estimator lgbm's best error=0.1080,	best estimator lgbm's best error=0.1080
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 4, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.3s,	estimator lgbm's best error=0.1080,	best estimator lgbm's best error=0.1080
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 5, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.4s,	estimator lgbm's best error=0.1062,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 6, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.4s,	estimator lgbm's best error=0.1062,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 7, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.5s,	estimator lgbm's best error=0.1062,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 8, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.5s,	estimator lgbm's best error=0.1062,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 9, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.7s,	estimator xgboost's best error=0.8713,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 10, current learner rf
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.8s,	estimator rf's best error=1.4230,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 11, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:50] {2466} INFO -  at 0.9s,	estimator lgbm's best error=0.1062,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:50] {2282} INFO - iteration 12, current learner rf
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.1s,	estimator rf's best error=1.0597,	best estimator lgbm's best error=0.1062
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 13, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.2s,	estimator lgbm's best error=0.1012,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 14, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.2s,	estimator xgboost's best error=0.4418,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 15, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.4s,	estimator lgbm's best error=0.1012,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 16, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.4s,	estimator xgboost's best error=0.3469,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 17, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.5s,	estimator lgbm's best error=0.1012,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 18, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.6s,	estimator xgboost's best error=0.1460,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 19, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.7s,	estimator xgboost's best error=0.1460,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 20, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.7s,	estimator xgboost's best error=0.1460,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 21, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:51] {2466} INFO -  at 1.9s,	estimator lgbm's best error=0.1012,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:51] {2282} INFO - iteration 22, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.1s,	estimator lgbm's best error=0.1012,	best estimator lgbm's best error=0.1012
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 23, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.2s,	estimator xgboost's best error=0.0854,	best estimator xgboost's best error=0.0854
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 24, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.3s,	estimator xgboost's best error=0.0854,	best estimator xgboost's best error=0.0854
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 25, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.3s,	estimator xgboost's best error=0.0854,	best estimator xgboost's best error=0.0854
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 26, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.4s,	estimator xgboost's best error=0.0800,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 27, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.5s,	estimator xgboost's best error=0.0800,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 28, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.6s,	estimator xgboost's best error=0.0800,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 29, current learner rf
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.7s,	estimator rf's best error=1.0597,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 30, current learner rf
    [flaml.automl.logger: 06-03 21:49:52] {2466} INFO -  at 2.9s,	estimator rf's best error=0.8248,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:52] {2282} INFO - iteration 31, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.1s,	estimator xgboost's best error=0.0800,	best estimator xgboost's best error=0.0800
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 32, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.2s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 33, current learner rf
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.4s,	estimator rf's best error=0.6586,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 34, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.5s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 35, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.6s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 36, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.7s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 37, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.8s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 38, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:53] {2466} INFO -  at 3.9s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:53] {2282} INFO - iteration 39, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.0s,	estimator xgboost's best error=0.0777,	best estimator xgboost's best error=0.0777
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 40, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.2s,	estimator xgboost's best error=0.0725,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 41, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.3s,	estimator lgbm's best error=0.1012,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 42, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.5s,	estimator xgboost's best error=0.0725,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 43, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.6s,	estimator xgboost's best error=0.0725,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 44, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.7s,	estimator xgboost's best error=0.0725,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 45, current learner rf
    [flaml.automl.logger: 06-03 21:49:54] {2466} INFO -  at 4.9s,	estimator rf's best error=0.6586,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:54] {2282} INFO - iteration 46, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.1s,	estimator xgboost's best error=0.0725,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 47, current learner rf
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.3s,	estimator rf's best error=0.5833,	best estimator xgboost's best error=0.0725
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 48, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.4s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 49, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.6s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 50, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.7s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 51, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:55] {2466} INFO -  at 5.9s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:55] {2282} INFO - iteration 52, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.0s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 53, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.1s,	estimator lgbm's best error=0.1012,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 54, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.3s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 55, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.4s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 56, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.5s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 57, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.6s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 58, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.7s,	estimator lgbm's best error=0.0864,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 59, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:56] {2466} INFO -  at 6.8s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:56] {2282} INFO - iteration 60, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.0s,	estimator lgbm's best error=0.0864,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 61, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.2s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 62, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.3s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 63, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.4s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 64, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.5s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 65, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.7s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 66, current learner rf
    [flaml.automl.logger: 06-03 21:49:57] {2466} INFO -  at 7.9s,	estimator rf's best error=0.5833,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:57] {2282} INFO - iteration 67, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.0s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 68, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.1s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 69, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.2s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 70, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.3s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 71, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.4s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 72, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.6s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 73, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.6s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 74, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.7s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 75, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.8s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 76, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:58] {2466} INFO -  at 8.9s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:58] {2282} INFO - iteration 77, current learner rf
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 9.2s,	estimator rf's best error=0.5833,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:59] {2282} INFO - iteration 78, current learner lgbm
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 9.4s,	estimator lgbm's best error=0.0825,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:59] {2282} INFO - iteration 79, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 9.5s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:59] {2282} INFO - iteration 80, current learner rf
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 9.6s,	estimator rf's best error=0.5833,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:59] {2282} INFO - iteration 81, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 9.7s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:49:59] {2282} INFO - iteration 82, current learner xgboost
    [flaml.automl.logger: 06-03 21:49:59] {2466} INFO -  at 10.0s,	estimator xgboost's best error=0.0700,	best estimator xgboost's best error=0.0700
    [flaml.automl.logger: 06-03 21:50:00] {2724} INFO - retrain xgboost for 0.1s
    [flaml.automl.logger: 06-03 21:50:00] {2727} INFO - retrained model: XGBRegressor(base_score=None, booster=None, callbacks=[],
                 colsample_bylevel=0.7653214871451958, colsample_bynode=None,
                 colsample_bytree=0.9380243961103967, device=None,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, feature_types=None, gamma=None,
                 grow_policy='lossguide', importance_type=None,
                 interaction_constraints=None, learning_rate=0.7708388938880748,
                 max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
                 max_delta_step=None, max_depth=0, max_leaves=32,
                 min_child_weight=4.311823268287431, missing=nan,
                 monotone_constraints=None, multi_strategy=None, n_estimators=4,
                 n_jobs=-1, num_parallel_tree=None, random_state=None, ...)
    [flaml.automl.logger: 06-03 21:50:00] {2009} INFO - fit succeeded
    [flaml.automl.logger: 06-03 21:50:00] {2010} INFO - Time taken to find the best model: 5.4128875732421875

