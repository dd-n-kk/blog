---
date:
  created: 2025-04-27
  updated: 2025-04-27

categories:
- Data preparation

tags:
- Polars

slug: polars-stratified-train-test-split
---

# Stratified train-test split with Polars

<a href="https://colab.research.google.com/github/dd-n-kk/notebooks/blob/main/blog/polars-stratified-train-test-split.ipynb" target="_parent">
    :simple-googlecolab: Colab notebook
</a>

<!-- more -->

## Preparations


```python
!uv pip install -Uq polars
```


```python
import numpy as np
import polars as pl
```


```python
_ = pl.Config(
    float_precision=3,
    fmt_str_lengths=200,
    fmt_table_cell_list_len=-1,
    tbl_cols=-1,
    tbl_rows=100,
    tbl_width_chars=100,
)

rng = np.random.default_rng(seed=777)
```

## Dummy data set


```python
labels = rng.choice(4, size=1000, p=[0.1, 0.2, 0.3, 0.4])
features = rng.standard_normal((1000, 2)) * 0.1 + labels[:, None]

data = (
    pl.concat(
        (
            pl.from_numpy(features, schema=["feat_1", "feat_2"]),
            pl.from_numpy(labels, schema=["label"]),
        ),
        how="horizontal",
    )
    .with_row_index(name="id")
)
```


```python
data.sample(5)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>id</th><th>feat_1</th><th>feat_2</th><th>label</th></tr><tr><td>u32</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>697</td><td>1.121</td><td>0.946</td><td>1</td></tr><tr><td>174</td><td>2.938</td><td>3.047</td><td>3</td></tr><tr><td>941</td><td>2.881</td><td>3.010</td><td>3</td></tr><tr><td>867</td><td>1.995</td><td>2.098</td><td>2</td></tr><tr><td>765</td><td>2.820</td><td>3.101</td><td>3</td></tr></tbody></table></div>




```python
data.get_column("label").value_counts().sort("label")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (4, 2)</small><table border="1" class="dataframe"><thead><tr><th>label</th><th>count</th></tr><tr><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>0</td><td>98</td></tr><tr><td>1</td><td>211</td></tr><tr><td>2</td><td>297</td></tr><tr><td>3</td><td>394</td></tr></tbody></table></div>



## Stratified train-test split

### Train split


```python
train_split = data.select(
    pl.all()
    .sample(fraction=0.9, shuffle=True, seed=777)
    .over("label", mapping_strategy="explode")
)
```


```python
train_split.sample(5)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>id</th><th>feat_1</th><th>feat_2</th><th>label</th></tr><tr><td>u32</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>860</td><td>2.946</td><td>2.852</td><td>3</td></tr><tr><td>876</td><td>1.893</td><td>1.875</td><td>2</td></tr><tr><td>327</td><td>3.035</td><td>3.023</td><td>3</td></tr><tr><td>657</td><td>-0.170</td><td>0.106</td><td>0</td></tr><tr><td>607</td><td>-0.188</td><td>0.241</td><td>0</td></tr></tbody></table></div>




```python
train_split.shape
```




    (898, 4)




```python
train_split.get_column("label").value_counts(normalize=True).sort("label")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (4, 2)</small><table border="1" class="dataframe"><thead><tr><th>label</th><th>proportion</th></tr><tr><td>i64</td><td>f64</td></tr></thead><tbody><tr><td>0</td><td>0.098</td></tr><tr><td>1</td><td>0.210</td></tr><tr><td>2</td><td>0.297</td></tr><tr><td>3</td><td>0.394</td></tr></tbody></table></div>



### Test (or validation) split


```python
test_split = data.join(train_split, on="id", how="anti")
```


```python
test_split.shape
```




    (102, 4)




```python
test_split.get_column("label").value_counts(normalize=True).sort("label")
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (4, 2)</small><table border="1" class="dataframe"><thead><tr><th>label</th><th>proportion</th></tr><tr><td>i64</td><td>f64</td></tr></thead><tbody><tr><td>0</td><td>0.098</td></tr><tr><td>1</td><td>0.216</td></tr><tr><td>2</td><td>0.294</td></tr><tr><td>3</td><td>0.392</td></tr></tbody></table></div>


