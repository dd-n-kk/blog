---
date:
  created: 2025-03-30
  updated: 2025-03-30

categories:
- Data preparation

tags:
- Polars
- Hugging Face
- "Series: GitHub repo issues dataset"

slug: make-huggingface-dataset-of-github-repo-issues
---

# Making a Hugging Face dataset of GitHub repo issues

This is Part II of my adaptation of the tutorial
[:simple-huggingface: Hugging Face NLP Course: Creating your own dataset][2].

Cleaned dataset on Hugging Face Hub: [:material-database: dd-n-kk/uv-github-issues][1]

<a href="https://colab.research.google.com/github/dd-n-kk/notebooks/blob/main/blog/make-huggingface-dataset-of-github-repo-issues.ipynb" target="_parent">
    :simple-googlecolab: Colab notebook
</a>

<!-- more -->

## Preparations


```python
# Set this to an empty string to avoid saving the dataset.
DATA_DIR = "uv-github-issues/"

# Set these to empty strings to avoid uploading the dataset.
REPO_ID = "dd-n-kk/uv-github-issues"
SECRET = "HF_TOKEN"
```


```python
!uv pip install --system -Uq polars
```


```python
import polars as pl
from polars import col
```


```python
SEED = 777
pl.set_random_seed(SEED)
_ = pl.Config(
    tbl_cols=-1,
    tbl_rows=100,
    tbl_width_chars=-1,
    float_precision=3,
    fmt_str_lengths=200,
    fmt_table_cell_list_len=-1,
)
```

The raw data were collected in [Part I][3].
To properly read a JSON Lines file into a Polars DataFrame,
`read_ndjson()` may need an increased `infer_schema_length`.


```python
issues_df = pl.read_ndjson("issues.jsonl", infer_schema_length=1000)
comments_df = pl.read_ndjson("comments.jsonl")
```

## Verifying completeness of data collection

First, we use a join to check whether we collected as many comments as recorded by GitHub
for each issue. It turns out that there are indeed a handful of mismatches.
However, according to the GitHub web pages (at the time of this post),
the numbers of the collected comments are more correct.


```python
(
    issues_df.lazy()
    .select("url", "html_url", "comments")
    .join(
        comments_df.lazy().group_by("issue_url").agg(pl.len().alias("collected_comments")),
        how="left",
        left_on="url",
        right_on="issue_url",
    )
    .filter(col("comments") != col("collected_comments"))
    .collect()
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (4, 4)</small><table border="1" class="dataframe"><thead><tr><th>url</th><th>html_url</th><th>comments</th><th>collected_comments</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>u32</td></tr></thead><tbody><tr><td>&quot;https://api.github.com/repos/astral-sh/uv/issues/6700&quot;</td><td>&quot;https://github.com/astral-sh/uv/issues/6700&quot;</td><td>9</td><td>8</td></tr><tr><td>&quot;https://api.github.com/repos/astral-sh/uv/issues/8635&quot;</td><td>&quot;https://github.com/astral-sh/uv/issues/8635&quot;</td><td>9</td><td>8</td></tr><tr><td>&quot;https://api.github.com/repos/astral-sh/uv/issues/8858&quot;</td><td>&quot;https://github.com/astral-sh/uv/pull/8858&quot;</td><td>2</td><td>1</td></tr><tr><td>&quot;https://api.github.com/repos/astral-sh/uv/issues/10230&quot;</td><td>&quot;https://github.com/astral-sh/uv/issues/10230&quot;</td><td>4</td><td>3</td></tr></tbody></table></div>



## Simplifying the issues DataFrame

- Discarded columns:
    - Null columns:
        - `active_lock_reason`
        - `performed_via_github_app`
        - `type`
    - Redundant or easily recoverable columns:
        - `assignee`
        - `comments_url`
        - `events_url`
        - `labels_url`
        - `repository_url`
        - `timeline_url`
    - Columns with little info:
        - `sub_issues_summary`

- Simplified columns:
    - Structs in `user`, `closed_by`, and `assignees` are replaced
      with their `login` fields.
    - Structs in `labels` are replaced with their `name` fields.
    - Structs in `milestone` are replaced with their `number` fields.
    - Values in `pull_request` are replaced with booleans
      of whether they were previously non-nulls.
      Also, the `merged_at` fields are extracted into a standalone column.
    - The `+1`, `-1`, `laugh`, `hooray`, `confused`, `heart`, `rocket`, and `eyes` fields
      of the Structs in `reactions` are extracted into standalone columns,
      with `+1` and `-1` renamed to `upvote` and `downvote`, respectively.
      The `reactions` column is then dropped.

- Since GitHub REST API uses a specific ISO 8601 format
  `⟨YYYY⟩-⟨MM⟩-⟨DD⟩T⟨hh⟩-⟨mm⟩-⟨ss⟩Z`,
  I decide to preserve the timestamp columns in this format (e.g. `updated_at`) as is.


```python
issues_df.select(pl.selectors.by_dtype(pl.Null)).columns
```




    ['type', 'active_lock_reason', 'performed_via_github_app']




```python
issues_df = (
    issues_df.lazy()
    .select(
        col("id"),
        col("node_id"),
        col("number"),
        col("url"),
        col("html_url"),

        col("title"),
        col("body"),

        col("user").struct.field("login").alias("user"),
        col("author_association"),

        col("labels").list.eval(pl.element().struct.field("name")),
        col("pull_request").is_not_null(),
        col("draft"),
        col("milestone").struct.field("number").alias("milestone"),

        col("state"),
        col("state_reason"),
        col("locked"),
        col("assignees").list.eval(pl.element().struct.field("login")),
        col("closed_by").struct.field("login").alias("closed_by"),

        col("created_at"),
        col("updated_at"),
        col("pull_request").struct.field("merged_at"),
        col("closed_at"),

        col("reactions").struct.field("+1").alias("upvote"),
        col("reactions").struct.field("-1").alias("downvote"),
        col("reactions").struct.field("laugh"),
        col("reactions").struct.field("hooray"),
        col("reactions").struct.field("confused"),
        col("reactions").struct.field("heart"),
        col("reactions").struct.field("rocket"),
        col("reactions").struct.field("eyes"),

        col("comments"),
    )
    .collect()
)
```

Some example queries on the simplified issues DataFrame:


```python
# Top 5 most upvoted issue titles:
(
    issues_df.select("title", "upvote")
    .sort("upvote", descending=True)
    .head(5)
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>title</th><th>upvote</th></tr><tr><td>str</td><td>i64</td></tr></thead><tbody><tr><td>&quot;Add option to upgrade all packages in the environment, e.g., `upgrade --all`&quot;</td><td>258</td></tr><tr><td>&quot;Using `uv run` as a task runner&quot;</td><td>257</td></tr><tr><td>&quot;Allow creating a `python` shim on `python install`&quot;</td><td>216</td></tr><tr><td>&quot;Add a command to activate the virtual environment, e.g., `uv shell`&quot;</td><td>213</td></tr><tr><td>&quot;Add a command to read and update (i.e., bump) the project version, e.g., `uv version`&quot;</td><td>177</td></tr></tbody></table></div>




```python
# Top 5 pull request authors:
(
    issues_df.filter(col("pull_request"))
    .get_column("user").value_counts()
    .sort("count", descending=True)
    .head(5)
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>user</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;charliermarsh&quot;</td><td>2728</td></tr><tr><td>&quot;zanieb&quot;</td><td>1342</td></tr><tr><td>&quot;konstin&quot;</td><td>804</td></tr><tr><td>&quot;renovate[bot]&quot;</td><td>472</td></tr><tr><td>&quot;BurntSushi&quot;</td><td>139</td></tr></tbody></table></div>



## Simplifying the comments DataFrame

- The null column `performed_via_github_app` is discarded.
- `user` and `reactions` are simplified the same way as the issues DataFrame.
- `issue_url` corresponds to `url` in the issues DataFrame,
  so it can serve as a join column. But for convenience I additionally create
  an `issue_number` column, which corresponds to `number` in the issues DataFrame.


```python
comments_df = (
    comments_df.lazy()
    .select(
        col("id"),
        col("node_id"),
        col("url"),
        col("html_url"),

        col("issue_url").str.extract(r"/(\d+)$").cast(pl.Int64).alias("issue_number"),
        col("issue_url"),

        col("body"),

        col("user").struct.field("login").alias("user"),
        col("author_association"),

        col("created_at"),
        col("updated_at"),

        col("reactions").struct.field("+1").alias("upvotes"),
        col("reactions").struct.field("-1").alias("downvotes"),
        col("reactions").struct.field("laugh"),
        col("reactions").struct.field("hooray"),
        col("reactions").struct.field("confused"),
        col("reactions").struct.field("heart"),
        col("reactions").struct.field("rocket"),
        col("reactions").struct.field("eyes"),
    )
    .collect()
)
```

## Organizing and sharing the Hugging Face dataset

I decide to organize the issues and comments DataFrames as separate sub-datasets
so that the users can choose to attach the comments data to each issue via joining.

### Creating the train-test splits

Semi-join is used to make sure the train-test splits of the comments subset
match correctly with the issues subset.


```python
TEST_SIZE = int(len(issues_df) * 0.2)

issues_df = issues_df.sample(len(issues_df), shuffle=True, seed=SEED)
issues_df_train = issues_df.head(len(issues_df) - TEST_SIZE)
issues_df_test = issues_df.tail(TEST_SIZE)
```


```python
comments_df_train = comments_df.join(
    issues_df_train, how="semi", left_on="issue_number", right_on="number"
)
comments_df_test = comments_df.join(
    issues_df_test, how="semi", left_on="issue_number", right_on="number"
)
```


```python
if DATA_DIR:
    import os

    os.makedirs(DATA_DIR, exist_ok=True)
    issues_df_train.write_ndjson(f"{DATA_DIR}/issues-train.jsonl")
    issues_df_test.write_ndjson(f"{DATA_DIR}/issues-test.jsonl")
    comments_df_train.write_ndjson(f"{DATA_DIR}/comments-train.jsonl")
    comments_df_test.write_ndjson(f"{DATA_DIR}/comments-test.jsonl")
```

### Configuring the subsets


```python
config_str = """---
configs:
- config_name: issues
  data_files:
  - split: train
    path: "issues-train.jsonl"
  - split: test
    path: "issues-test.jsonl"
  default: true
- config_name: comments
  data_files:
  - split: train
    path: "comments-train.jsonl"
  - split: test
    path: "comments-test.jsonl"
---
"""

if DATA_DIR:
    with open(f"{DATA_DIR}/README.md", "w") as file:
        file.write(config_str)
```

### Uploading and downloading the dataset


```python
if REPO_ID and SECRET:
    from google.colab import userdata
    !huggingface-cli upload uv-github-issues uv-github-issues/ . --repo-type dataset --token={userdata.get(SECRET)}
```

    Consider using `hf_transfer` for faster uploads. This solution comes with some limitations. See https://huggingface.co/docs/huggingface_hub/hf_transfer for more details.
    Start hashing 5 files.
    Finished hashing 5 files.
    comments-train.jsonl:   0% 0.00/28.5M [00:00<?, ?B/s]
    Upload 2 LFS files:   0% 0/2 [00:00<?, ?it/s][A
    
    comments-train.jsonl:  13% 3.60M/28.5M [00:00<00:00, 35.8MB/s]
    
    issues-train.jsonl:   2% 344k/18.8M [00:00<00:05, 3.32MB/s][A[A
    
    comments-train.jsonl:  25% 7.19M/28.5M [00:00<00:02, 7.14MB/s]
    
    comments-train.jsonl:  31% 8.95M/28.5M [00:01<00:02, 7.98MB/s]
    
    comments-train.jsonl:  39% 11.0M/28.5M [00:01<00:01, 8.91MB/s]
    
    comments-train.jsonl:  51% 14.5M/28.5M [00:01<00:01, 12.9MB/s]
    
    comments-train.jsonl:  78% 22.1M/28.5M [00:02<00:00, 13.4MB/s]
    
    comments-train.jsonl: 100% 28.5M/28.5M [00:03<00:00, 7.67MB/s]
    issues-train.jsonl: 100% 18.8M/18.8M [00:03<00:00, 5.02MB/s]
    
    Upload 2 LFS files:  50% 1/2 [00:04<00:04,  4.15s/it][A
    Upload 2 LFS files: 100% 2/2 [00:05<00:00,  2.64s/it]
    https://huggingface.co/datasets/dd-n-kk/uv-github-issues/tree/main/.



```python
!uv pip install --system -q datasets
```


```python
from datasets import load_dataset
```


```python
issues_ds = load_dataset("dd-n-kk/uv-github-issues")
```


    README.md:   0%|          | 0.00/299 [00:00<?, ?B/s]



    issues-train.jsonl:   0%|          | 0.00/18.8M [00:00<?, ?B/s]



    issues-test.jsonl:   0%|          | 0.00/4.89M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/9963 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/2490 [00:00<?, ? examples/s]



```python
comments_ds = load_dataset("dd-n-kk/uv-github-issues", "comments")
```


    comments-train.jsonl:   0%|          | 0.00/28.5M [00:00<?, ?B/s]



    comments-test.jsonl:   0%|          | 0.00/7.18M [00:00<?, ?B/s]



    Generating train split:   0%|          | 0/27293 [00:00<?, ? examples/s]



    Generating test split:   0%|          | 0/6682 [00:00<?, ? examples/s]


[1]: https://huggingface.co/datasets/dd-n-kk/uv-github-issues
[2]: https://huggingface.co/learn/nlp-course/en/chapter5/5
[3]: get-github-repo-issues-comments.md
