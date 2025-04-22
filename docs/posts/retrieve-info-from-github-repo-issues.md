---
date:
  created: 2025-04-22
  updated: 2025-04-22

categories:
- Deep learning

tags:
- NLP
- Polars
- Hugging Face
- "Series: GitHub repo issues dataset"

slug: retrieve-info-from-github-repo-issues
---

# Retrieving information from a GitHub repo issues dataset

This is Part III of my adaptation of
:simple-huggingface: Hugging Face NLP Course: [Creating your own dataset][1].
It consists of several parts:

1. Creating a corpus of issue-comment pairs from the previously prepared dataset.
2. Embedding each issue-comment pair into dense vectors for similarity search.
3. Building a Faiss index of the embeddings to speed up querying.
4. Using a reranker to pick the best entries from the top-k similarity search results.

<a href="https://colab.research.google.com/github/dd-n-kk/notebooks/blob/main/blog/retrieve-info-from-github-repo-issues.ipynb" target="_parent">
    :simple-googlecolab: Colab notebook
</a>

<!-- more -->

## Preparations


```python
!uv pip install -Uq polars
!uv pip install -q datasets faiss-cpu
```


```python
from collections.abc import Sequence

import faiss
import numpy as np
import polars as pl
import torch as tc
from datasets import load_dataset
from numpy.typing import NDArray
from polars import col
from torch.nn import functional as F
from tqdm.auto import trange
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
```


```python
_ = pl.Config(
    float_precision=3,
    fmt_str_lengths=200,
    fmt_table_cell_list_len=-1,
    tbl_cols=-1,
    tbl_rows=100,
    tbl_width_chars=-1,
)
```

## Preparing the corpus

We directly download the `dd-n-kk/uv-github-issues` dataset prepared in [Part II][2].


```python
%%capture
issues = load_dataset("dd-n-kk/uv-github-issues", "issues")
comments = load_dataset("dd-n-kk/uv-github-issues", "comments")
```

The train-test splits are merged because all processed entries will be used for querying.


```python
issues_df = pl.concat([issues["train"].to_polars(), issues["test"].to_polars()])
comments_df = pl.concat([comments["train"].to_polars(), comments["test"].to_polars()])
```

### Issues

I decide to remove:

- Issues with null bodies.
- Issues created by bots, because they usually contain little info.
- Pull requests not yet merged, for they often contain suggestions not yet adopted.


```python
issues_df = issues_df.filter(
    col("body").is_not_null()
    & (~col("user").str.contains("[bot]", literal=True))
    & (~col("pull_request") | col("merged_at").is_not_null())
)
```

A crude word count reveals a problem: A small number of issues are extremely long.


```python
q = (
    issues_df.select(
        "html_url", "title", col("body").str.split(" ").list.len().alias("n_words")
    )
    .sort("n_words", descending=True)
)
q.get_column("n_words").describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>value</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>10313.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>152.041</td></tr><tr><td>&quot;std&quot;</td><td>424.822</td></tr><tr><td>&quot;min&quot;</td><td>1.000</td></tr><tr><td>&quot;25%&quot;</td><td>21.000</td></tr><tr><td>&quot;50%&quot;</td><td>62.000</td></tr><tr><td>&quot;75%&quot;</td><td>151.000</td></tr><tr><td>&quot;max&quot;</td><td>12590.000</td></tr></tbody></table></div>



From the GitHub web pages we can see that they contain long debug outputs,
which are unlikely to help answer user questions.


```python
q.head(5)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 3)</small><table border="1" class="dataframe"><thead><tr><th>html_url</th><th>title</th><th>n_words</th></tr><tr><td>str</td><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;https://github.com/astral-sh/uv/issues/6443&quot;</td><td>&quot;`uv sync` freezes infinitely at the container root&quot;</td><td>12590</td></tr><tr><td>&quot;https://github.com/astral-sh/uv/issues/5742&quot;</td><td>&quot;Allow `uv sync --no-build-isolation`&quot;</td><td>11862</td></tr><tr><td>&quot;https://github.com/astral-sh/uv/issues/5046&quot;</td><td>&quot;Bad resolver error for `colabfold[alphafold]==1.5.5` on python 3.11&quot;</td><td>11764</td></tr><tr><td>&quot;https://github.com/astral-sh/uv/issues/7183&quot;</td><td>&quot;Improve Python version resolution UI&quot;</td><td>11316</td></tr><tr><td>&quot;https://github.com/astral-sh/uv/issues/2062&quot;</td><td>&quot;uv pip and python -m pip resolve different versions of tensorflow in pyhf developer environment&quot;</td><td>8819</td></tr></tbody></table></div>



To shorten these issues, I:

- Replace Markdown fenced code blocks containing too many characters with `[CODE]`.
- Replace each HTML element `<detail>` with `[DETAIL]`.
- Replace HTML comments `<!-- ... -->` with `[COMMENT]`.
- Remove trailing whitespaces.


```python
issues_df = (
    issues_df.lazy()
    .select(
        "number",
        col("html_url").alias("issue_url"),
        "title",
        (
            col("body").str.replace_all(r"\s*[\r\n]", "\n")
            .str.replace_all(r"(?s)<details>.*?</details>", "[DETAILS]")
            .str.replace_all(r"(?s)<!--.*?-->", "[COMMENT]")
            .str.replace_all(r"```(?:[^`]|`[^`]|``[^`]){768,}```", "[CODE]")
            .str.replace_all(r"~~~(?:[^~]|~[^~]|~~[^~]){768,}~~~", "[CODE]")
        ),
    )
    .collect()
)
```

An issue body containing long debug outputs now looks like this:


```python
print(issues_df.filter(col("number") == 6443).item(0, "body"))
```

    To reproduce, try building the following Dockerfile (remove `sudo` if your docker is rootless):
    ```
    cat <<EOF | sudo BUILDKIT_PROGRESS=plain docker build -
    FROM library/python:3.11
    RUN pip install 'uv == 0.3.1' \
        && printf >pyproject.toml '\
          [project]\n\
          dependencies = ["django ~= 4.2"]\n\
          name = "demo"\n\
          version = "0.1.0"\n\
          requires-python = ">=3.11.7"\n\
        '\
        && uv lock
    RUN uv sync -vv
    EOF
    ```
    This is not specific to Django, according to my experiments.
    The build freezes after `uv_build::run_python_script script="get_requires_for_build_editable", python_version=3.11.9` verbose log, keeping a high CPU load for a few minutes.
    [DETAILS]
    However, this only happens if I build this at the root of filesystem. Adding `WORKDIR /home` before installation recovers everything, the build completes in seconds.
    [DETAILS]
    


The word counts are now subtantially reduced.


```python
issues_df.get_column("body").str.split(" ").list.len().describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>value</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>10313.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>81.815</td></tr><tr><td>&quot;std&quot;</td><td>105.018</td></tr><tr><td>&quot;min&quot;</td><td>1.000</td></tr><tr><td>&quot;25%&quot;</td><td>19.000</td></tr><tr><td>&quot;50%&quot;</td><td>51.000</td></tr><tr><td>&quot;75%&quot;</td><td>110.000</td></tr><tr><td>&quot;max&quot;</td><td>3260.000</td></tr></tbody></table></div>



### Comments

The comments dataset is processed similarly:
Bot comments are removed and long code blocks are snipped.


```python
comments_df = comments_df.filter(~col("user").str.contains("[bot]", literal=True))
```


```python
comments_df.get_column("body").str.split(" ").list.len().describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>value</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>33789.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>69.231</td></tr><tr><td>&quot;std&quot;</td><td>454.410</td></tr><tr><td>&quot;min&quot;</td><td>1.000</td></tr><tr><td>&quot;25%&quot;</td><td>12.000</td></tr><tr><td>&quot;50%&quot;</td><td>25.000</td></tr><tr><td>&quot;75%&quot;</td><td>55.000</td></tr><tr><td>&quot;max&quot;</td><td>39896.000</td></tr></tbody></table></div>




```python
comments_df = (
    comments_df.lazy()
    .select(
        "issue_number",
        col("html_url").alias("comment_url"),
        (
            col("body").str.replace_all(r"\s*[\r\n]", "\n")
            .str.replace_all(r"(?s)<details>.*?</details>", "[DETAILS]")
            .str.replace_all(r"(?s)<!--.*?-->", "[COMMENT]")
            .str.replace_all(r"```(?:[^`]|`[^`]|``[^`]){768,}```", "[CODE]")
            .str.replace_all(r"~~~(?:[^~]|~[^~]|~~[^~]){768,}~~~", "[CODE]")
            .alias("comment_body")
        ),
    )
    .collect()
)
```


```python
comments_df.get_column("comment_body").str.split(" ").list.len().describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>value</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>33789.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>43.568</td></tr><tr><td>&quot;std&quot;</td><td>62.287</td></tr><tr><td>&quot;min&quot;</td><td>1.000</td></tr><tr><td>&quot;25%&quot;</td><td>12.000</td></tr><tr><td>&quot;50%&quot;</td><td>24.000</td></tr><tr><td>&quot;75%&quot;</td><td>51.000</td></tr><tr><td>&quot;max&quot;</td><td>1934.000</td></tr></tbody></table></div>



### Joining

We are ready to create the corpus.
I've considered combining each issue with all associated comments,
but that may require an embedding model with a very large context length.
Therefore, I use left join to create issue-comment pairs
while preserving issues with no comment.

URLs are also collected for convenient lookups.


```python
corpus_df = (
    issues_df.lazy()
    .join(comments_df.lazy(), how="left", left_on="number", right_on="issue_number")
    .select(
        (
            pl.when(col("comment_url").is_null())
            .then(col("issue_url"))
            .otherwise("comment_url")
            .alias("url")
        ),
        (
            pl.when(col("comment_body").is_null())
            .then(
                pl.format("Issue {}: {}\n\n{}", col("number"), col("title"), col("body"))
            )
            .otherwise(
                pl.format(
                    "Issue {}: {}\n\n{}\n\nComment:\n{}",
                    col("number"),
                    col("title"),
                    col("body"),
                    col("comment_body"),
                )
            )
            .alias("text")
        ),
    )
    .sort("url")
    .collect()
)
```


```python
assert corpus_df.get_column("url").n_unique() == len(corpus_df)
```


```python
corpus_df.get_column("text").str.split(" ").list.len().describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 2)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>value</th></tr><tr><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>35278.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>162.174</td></tr><tr><td>&quot;std&quot;</td><td>149.027</td></tr><tr><td>&quot;min&quot;</td><td>3.000</td></tr><tr><td>&quot;25%&quot;</td><td>68.000</td></tr><tr><td>&quot;50%&quot;</td><td>128.000</td></tr><tr><td>&quot;75%&quot;</td><td>213.000</td></tr><tr><td>&quot;max&quot;</td><td>3312.000</td></tr></tbody></table></div>



## Embedding issue-comment pairs

!!! warning "&#8203;"
    Running this section likely requires GPU.

I pick [`BAAI/bge-m3`][3] as the pretrained embedder.
It is based on [`FacebookAI/xlm-roberta-large`][4].
It is reasonably sized for a Colab T4 GPU, has a long enough context length of 8192,
and is versatile and efficient.


```python
corpus = corpus_df.get_column("text").to_list()
```


```python
%%capture
EMB_CKPT = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(EMB_CKPT)
embedder = AutoModel.from_pretrained(EMB_CKPT)
```

To reduce unncessary padding, the padding length is determined batch by batch.
Also, the encoded entries are batched in decreasing lengths,
so that the batch maximum length accomodates the entries efficiently.
We do have to restore the embeddings to the original order.


```python
def embed(
    texts: Sequence[str],
    *,
    tokenizer,
    embedder,
    batch_size: int,
    context_len: int,
    device=None,
    use_half: bool = True,
) -> NDArray:
    no_tqdm = len(texts) < batch_size
    if device is None:
        device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    encodings = []
    for i in trange(0, len(texts), batch_size, desc="Tokenization", disable=no_tqdm):
        # No padding now; pad within each embedder input batch.
        batch = tokenizer(
            texts[i : i + batch_size], truncation=True, max_length=context_len
        )

        # dict[list] -> list[dict]
        ## Just one way to conform to `tokenizer.pad()`.
        encodings.extend(dict(zip(batch, vals)) for vals in zip(*batch.values()))

    # Sort by token count in descending order to reduce padding.
    # Keep the sorted index to restore the original order later.
    ## Reverse view > element-wise negative (https://stackoverflow.com/a/16486305)
    sorted_index = np.argsort([len(x["input_ids"]) for x in encodings])[::-1]
    encodings = [encodings[i] for i in sorted_index]

    embedder = embedder.to(device).eval()
    # Using float16 only on GPU.
    if device.type == "cuda" and use_half:
        embedder = embedder.half()

    embeddings = []
    with tc.inference_mode():
        for i in trange(
            0,
            len(encodings),
            batch_size,
            desc="Embedding",
            disable=len(encodings) < batch_size,
        ):
            # Within-batch padding
            ## `BatchEncoding` has method `to()`.
            padded = tokenizer.pad(
                encodings[i : i + batch_size],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # [CLS] pooling with normalization
            embeddings.append(
                F.normalize(embedder(**padded).last_hidden_state[:, 0], dim=-1)
                .cpu()
                .numpy()
            )

    # Merge, cast to float32 (for Faiss), and restore original order.
    return np.concatenate(embeddings, 0, dtype=np.float32)[np.argsort(sorted_index)]
```

The embedding process takes about 10 minutes in a Colab T4 GPU runtime.


```python
embeddings = embed(
    corpus,
    tokenizer=tokenizer,
    embedder=embedder,
    batch_size=32,
    context_len=4096,
    use_half=True,
)
```


    Tokenization:   0%|          | 0/1103 [00:00<?, ?it/s]



    Embedding:   0%|          | 0/1103 [00:00<?, ?it/s]


    You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.



```python
embeddings.shape
```




    (35278, 1024)



We can now create some example user questions about `astral-sh/uv` as queries.


```python
queries = [
    "What is the difference between `uv pip install` and `uv add`?",
    "How to update Python in my venv to the latest version?",
    "How to install the CPU version of PyTorch?",
    "Can I add a package dependency without version requirement?",
    "What does the `.python-version` file do?",
]
```


```python
q_embeddings = embed(
    queries,
    tokenizer=tokenizer,
    embedder=embedder,
    batch_size=8,
    context_len=512,
    use_half=True,
)
```


```python
%timeit (q_embeddings @ embeddings.T)
```

    55.6 ms ± 1.6 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


The higher the inner product of a query embedding and an issue-comment pair embedding,
the more similar they should be,
and the more likely the issue-comment pair contains an answer to the question.


```python
result_indexes = (q_embeddings @ embeddings.T).argmax(-1).tolist()
```


```python
def display_query_results(queries, corpus_df, indexes):
    for query, i in zip(queries, indexes):
        print(f"Query: {query}\n")
        print(f"Result URL: {corpus_df.get_column('url')[i]}\n")
        print(f"Result:\n{corpus_df.get_column('text')[i]}\n")
        print(f"==================================================\n")
```

Even though only some of the comments directly answer the questions,
all the retrieved issues are indeed highly relevant.


```python
display_query_results(queries, corpus_df, result_indexes)
```

    Query: What is the difference between `uv pip install` and `uv add`?
    
    Result URL: https://github.com/astral-sh/uv/issues/9219#issuecomment-2613016513
    
    Result:
    Issue 9219: What's the difference between `uv pip install` and `uv add`
    
    [COMMENT]
    I've been using `uv` for a while and I really enjoy it. I keep stumbling onto the same confusion though. I never quite know whether do to:
    ```sh
    uv init
    uv venv
    uv add polars marimo
    uv run hello.py
    ```
    or
    ```sh
    uv init
    uv venv
    source .venv/bin/activate
    pip install polars marimo
    python hello.py
    ```
    are these two above equivalent?
    ---
    also are these two equivalent?
    ```
    uv add polars
    ```
    ```
    uv pip install polars
    ```
    
    Comment:
    1. `pip install` will only install into its own environment, `uv pip install` can target other environments.
    2. `uv run script.py` will activate a virtual environment if necessary, and read PEP 723 inline metadata or sync your project if necessary, then call `python script.py` — the latter just uses your current environment as-is.
    
    ==================================================
    
    Query: How to update Python in my venv to the latest version?
    
    Result URL: https://github.com/astral-sh/uv/issues/11317#issuecomment-2643263625
    
    Result:
    Issue 11317: How to update `uv` managed Python version for applications?
    
    ### Question
    I have Python installed via uv's `uv python install 3.x`. The project is an project/application (so not a library).
    Let's say there is a new Python (standalone) release, e.g.
    3.12.9 -> 3.13.2
    3.13.1 -> 3.13.2
    It there a way to 'update' the standalone Python from uv, also replacing the existing `.venv`?
    My current solution is to install the new python with `uv python install 3.x`, deleting the .venv and recreating it with the target python version (and delete the `.python-version` file). This works fine, but is a rather tedious process.
    I would love to have a project scoped command like `uv sync --update-python-to 3.13.2`  (handling also the installation of the standalone-python).
    Similar possibilities:
    - `uv python update` (either updating existing standalone python versions on patch level AND/OR updating project .venv on python version patch level)
    - `uv python update 3.13` (updating project .venv to latest specified python version)
    I guess there are smarter persons, having better ideas how to handle these commands. Is there something in uv right now, that I'm misssing?
    ### Platform
    _No response_
    ### Version
    0.5.29
    
    Comment:
    Dang, didn't found that one. Thx!
    
    ==================================================
    
    Query: How to install the CPU version of PyTorch?
    
    Result URL: https://github.com/astral-sh/uv/issues/11079#issuecomment-2624678978
    
    Result:
    Issue 11079: Add pytorch documentation section on how to install for Intel GPUs
    
    ### Summary
    Pytorch 2.6 [adds support for Intel GPUs](https://pytorch.org/docs/main/notes/get_start_xpu.html). The current [uv pytorch install docs](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) include instructions for:
    - CPU only
    - Various CUDA versions
    - ROCm
    But not yet for Intel GPUs.
    My current attempt at mimicking the docs for existing GPUs is
    ```pyproject.toml
    # pyproject.toml
    [project]
    name = "pytorch-intel"
    version = "0.1.0"
    description = "Add your description here"
    readme = "README.md"
    requires-python = ">=3.12"
    dependencies = ["torch>=2.6.0"]
    [[tool.uv.index]]
    name = "pytorch-intel-gpu"
    url = "https://download.pytorch.org/whl/xpu"
    explicit = true
    [tool.uv.sources]
    torch = [{ index = "pytorch-intel-gpu", marker = "platform_system == 'Linux'" }]
    ```
    ```python
    # hello.py
    import torch
    print(torch.xpu.is_available())
    ```
    Running `uv run hello.py` produces
    ```sh
      × No solution found when resolving dependencies for split (sys_platform
      │ == 'linux'):
      ╰─▶ Because there is no version of pytorch-triton-xpu{platform_machine
          == 'x86_64' and sys_platform == 'linux'}==3.2.0 and torch==2.6.0+xpu
          depends on pytorch-triton-xpu{platform_machine == 'x86_64'
          and sys_platform == 'linux'}==3.2.0, we can conclude that
          torch==2.6.0+xpu cannot be used.
          And because only the following versions of torch{sys_platform ==
          'linux'} are available:
              torch{sys_platform == 'linux'}<2.6.0
              torch{sys_platform == 'linux'}==2.6.0+xpu
          and your project depends on torch{sys_platform == 'linux'}>=2.6.0, we
          can conclude that your project's requirements are unsatisfiable.
    ```
    ### Example
    _No response_
    
    Comment:
    Yes, this worked for me!
    I got a warning from pytorch that it couldn't initialize numpy. Adding numpy to the pyproject fixed the warning. 
    
    ==================================================
    
    Query: Can I add a package dependency without version requirement?
    
    Result URL: https://github.com/astral-sh/uv/issues/6476#issuecomment-2305937941
    
    Result:
    Issue 6476: Allow adding a dependency with no version constraint
    
    `uv add` adds a lower bound version constraint by default. For example, calling `uv add requests` currently adds the dependency `"requests>=2.32.3"` to `pyproject.toml`.
    It's possible to have uv use different upper and/or lower bounds, but I could not find a way to add a dependency without any version constraint at all. For example, I want the dependency `"requests"` added to `pyproject.toml` without any bounds.
    The current default behavior is fine, but I think there should be some way to add an unconstrained dependency, either with a global configuration setting that changes the default behavior, or with a command-line option for `uv add`.
    Apologies if I missed something in the documentation and this is already possible. (Of course I could edit `pyproject.toml` manually, but it doesn't seem like that should be necessary.)
    Thanks for the great work on a fantastic project!
    
    Comment:
    That seems ok to me.
    
    ==================================================
    
    Query: What does the `.python-version` file do?
    
    Result URL: https://github.com/astral-sh/uv/issues/8920#issuecomment-2465005344
    
    Result:
    Issue 8920: Purpose of .python-version?
    
    [COMMENT]
    Currently on `uv init` a project boilerplate created includes the file `.python-version`.
    What is the purpose of this file, if there are constraints in `pyproject.toml` and `uv.lock`?
    Is it just for compatibility with tools like pyenv or is there more to it?
    I really like how clean project directory became after moving a lot of stuff (like `requirements.*`) to `pyproject.toml`. So wonder, if `.python-version` is really required in two senses:
    1. Is it required for the projects now? I tested by deleting `.python-version`, and everything works as expected and this file is not recreated with commands like `uv run`, `uv sync` and `uv lock`
    2. Since a lot of constraints/pinned versions of things are now in `pyproject.toml` and `uv.lock`, maybe if `.python-version` still plays some important role, its function can be moved to `pyproject.toml` or `uv.lock` in future?
    Asking as a 5-year user of `pyenv` and `.python-version`. They are great, but I don't really miss them.
    Also, couldn't find peps related to this file. 
    
    Comment:
    Related - https://github.com/astral-sh/uv/issues/8247
    A `.python-version` file is not strictly required, but it's useful to have it when developing a project as it allows you to specify the exact Python version you are using to do development. It's different to the `requires-python` field  - that is the range of Python versions supported by your project.
    
    
    ==================================================
    


## Building a Faiss index to improve query speed

[Faiss][5] is a library that can create indexes to speed up similarity search
among dense vector embeddings, often at very little cost of accuracy.

I use an inverted file index with inner product as metric.
The `nlist` is the number of partitions made (in the form of inverted lists)
in the embedding space,
and the `nprob` is the number of partitions examined per query.
They are set roughly according to the [guideilnes][6].


```python
D = embeddings.shape[-1]
nlist = 2048
nprob = 16
```


```python
quantizer = faiss.IndexFlatIP(D)
faiss_index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
```

Building the index takes less than 1 minute.


```python
faiss_index.train(embeddings)
faiss_index.add(embeddings)
```


```python
faiss_index.is_trained, faiss_index.ntotal, faiss_index.nprobe
```




    (True, 35278, 1)




```python
faiss_index.nprobe = nprob
```

The index improves query time by a factor of $\sim 20$,
and happens to get exactly the same results from our example queries.


```python
%timeit faiss_index.search(q_embeddings, k=1)
```

    2.71 ms ± 69 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
metrics, faiss_result_indexes = faiss_index.search(q_embeddings, k=1)
result_indexes, faiss_result_indexes.reshape(-1).tolist()
```




    ([25352, 2833, 2230, 18026, 24623], [25352, 2833, 2230, 18026, 24623])



## Using a Reranker to improve query results

A reranker is supposed to evaluate relevance between embeddings
more accurately but slowly than distance metrics.

Here I use [`BAAI/bge-reranker-v2-m3`][7], the matching reranker of `BAAI/bge-m3`.


```python
%%capture
RRK_CKPT = "BAAI/bge-reranker-v2-m3"
rerank_tokenizer = AutoTokenizer.from_pretrained(RRK_CKPT)
reranker = AutoModelForSequenceClassification.from_pretrained(RRK_CKPT)
```


```python
def rerank(
    query: str,
    corpus: Sequence[str],
    indexes: Sequence[int],
    tokenizer,
    reranker,
    context_len: int,
    device=None,
):
    if device is None:
        device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

    pairs = [[query, corpus[i]] for i in indexes]
    inputs = tokenizer(
        pairs, padding=True, truncation=True, max_length=context_len, return_tensors="pt"
    ).to(device)

    reranker = reranker.to(device).eval()
    with tc.inference_mode():
        logits = reranker(**inputs).logits.reshape(-1).cpu().numpy()

    return np.array(indexes)[(-logits).argsort()]
```

First, issue-comment pairs of top-10 similarities are retrieved from the Faiss index.
Then, the reranker reorders them in decreasing scores.


```python
metrics, faiss_result_indexes = faiss_index.search(q_embeddings, k=10)
```


```python
reranked_indexes = np.array(
    [
        rerank(q, corpus, i, rerank_tokenizer, reranker, context_len=4096)
        for q, i in zip(queries, faiss_result_indexes)
    ]
)
```

With our examples, the final top-1 results are quite similar.
But the reranked results for question 1 and 3 are arguably more complete.


```python
faiss_result_indexes
```




    array([[25352, 25339, 25344, 25340, 25351, 25350, 25345, 25343, 25346,
            25341],
           [ 2833,  2832, 22932, 25560, 20466, 25559, 20467, 20468, 20469,
            20464],
           [ 2230,  2231,  6440,  6458,  2229,  6457,  6455, 16575,  6453,
             6437],
           [18026, 18024, 18025, 23750,  4664, 12986, 13010, 16315,  4665,
            13004],
           [24623, 24624, 33204,  8035,  4781, 25094, 20774,  8034,  3762,
            15749]])




```python
reranked_indexes
```




    array([[25339, 25352, 25340, 25344, 25341, 25350, 25351, 25346, 25345,
            25343],
           [ 2833,  2832, 25559, 20466, 25560, 20467, 20464, 22932, 20468,
            20469],
           [ 6458,  6440,  6453,  6457,  6455,  6437,  2229,  2230,  2231,
            16575],
           [18026, 18024, 18025,  4665,  4664, 23750, 16315, 12986, 13010,
            13004],
           [24623, 25094, 15749, 24624,  4781,  8034,  3762,  8035, 33204,
            20774]])




```python
display_query_results(queries, corpus_df, reranked_indexes[:, 0].reshape(-1).tolist())
```

    Query: What is the difference between `uv pip install` and `uv add`?
    
    Result URL: https://github.com/astral-sh/uv/issues/9219#issuecomment-2485573603
    
    Result:
    Issue 9219: What's the difference between `uv pip install` and `uv add`
    
    [COMMENT]
    I've been using `uv` for a while and I really enjoy it. I keep stumbling onto the same confusion though. I never quite know whether do to:
    ```sh
    uv init
    uv venv
    uv add polars marimo
    uv run hello.py
    ```
    or
    ```sh
    uv init
    uv venv
    source .venv/bin/activate
    pip install polars marimo
    python hello.py
    ```
    are these two above equivalent?
    ---
    also are these two equivalent?
    ```
    uv add polars
    ```
    ```
    uv pip install polars
    ```
    
    Comment:
    ``uv add`` choose universal or cross-platform dependencies , and ``uv add`` is a project API.
    https://docs.astral.sh/uv/concepts/projects/
    This is my understanding, but the more correct interpretation should be based on the documentation and the uv team's explanation.
    > Suppose a dependency has versions 1.0.0 and 1.1.0 on Windows, but versions 1.0.0, 1.1.0, and 1.2.0 on Linux. If you're using uv on Linux, uv pip install would typically install the latest version (1.2.0), while uv add would select version 1.1.0, ensuring compatibility across Windows and Linux.
    The explanation in the document should be here:
    * uv add
    > uv's lockfile (uv.lock) is created with a universal resolution and is portable across platforms. This ensures that dependencies are locked for everyone working on the project, regardless of operating system, architecture, and Python version. The uv lockfile is created and modified by project commands such as uv lock, uv sync, and uv add.
    https://docs.astral.sh/uv/concepts/resolution/#universal-resolution
    * uv pip install
    > By default, uv tries to use the latest version of each package. For example, uv pip install flask>=2.0.0 will install the latest version of Flask, e.g., 3.0.0. If flask>=2.0.0 is a dependency of the project, only flask 3.0.0 will be used. This is important, for example, because running tests will not check that the project is actually compatible with its stated lower bound of flask 2.0.0.
    https://docs.astral.sh/uv/concepts/resolution/#resolution-strategy
    
    
    ==================================================
    
    Query: How to update Python in my venv to the latest version?
    
    Result URL: https://github.com/astral-sh/uv/issues/11317#issuecomment-2643263625
    
    Result:
    Issue 11317: How to update `uv` managed Python version for applications?
    
    ### Question
    I have Python installed via uv's `uv python install 3.x`. The project is an project/application (so not a library).
    Let's say there is a new Python (standalone) release, e.g.
    3.12.9 -> 3.13.2
    3.13.1 -> 3.13.2
    It there a way to 'update' the standalone Python from uv, also replacing the existing `.venv`?
    My current solution is to install the new python with `uv python install 3.x`, deleting the .venv and recreating it with the target python version (and delete the `.python-version` file). This works fine, but is a rather tedious process.
    I would love to have a project scoped command like `uv sync --update-python-to 3.13.2`  (handling also the installation of the standalone-python).
    Similar possibilities:
    - `uv python update` (either updating existing standalone python versions on patch level AND/OR updating project .venv on python version patch level)
    - `uv python update 3.13` (updating project .venv to latest specified python version)
    I guess there are smarter persons, having better ideas how to handle these commands. Is there something in uv right now, that I'm misssing?
    ### Platform
    _No response_
    ### Version
    0.5.29
    
    Comment:
    Dang, didn't found that one. Thx!
    
    ==================================================
    
    Query: How to install the CPU version of PyTorch?
    
    Result URL: https://github.com/astral-sh/uv/issues/1497#issuecomment-2102236399
    
    Result:
    Issue 1497: Cannot install the CPU version of torch
    
    I tried to install the CPU version of torch but could not.
    ```bash
    uv pip install torch==2.1.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html
    # and next command gives the same result.
    uv pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    ```
    ```log
      × No solution found when resolving dependencies:
      ╰─▶ Because there is no version of torch==2.1.0+cpu and you require torch==2.1.0+cpu, we can conclude that the requirements are unsatisfiable.
    ```
    uv version: v0.1.2
    Python 3.11.7
    Ubuntu 20.4
    X86_64 Architecture
    ---
    By the way, the install of the GPU version of torch is successful.
    ```bash
    uv pip install torch==2.1.0
    # success
    ```
    
    Comment:
    Perhaps it's due to a lack of supported variants that they omitted the local identifier there? (_[here upstream PyTorch states they don't intend to publish `aarch64` variants with GPU accel](https://github.com/pytorch/pytorch/issues/110791#issuecomment-1753315240)_)
    ```bash
    # Torch 2.3.0 + Python 3.12 Linux x86_64/aarch64 wheels
    # +cpu
    # https://download.pytorch.org/whl/cpu/torch/
    torch-2.3.0+cpu-cp312-cp312-linux_x86_64.whl
    # No local identifier for arm64:
    torch-2.3.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
    # +cu121
    # https://download.pytorch.org/whl/cu121/torch/
    torch-2.3.0+cu121-cp312-cp312-linux_x86_64.whl
    ```
    ## Reference
    Using Docker from an AMD64 machine to test ARM64 environment:
    ```console
    # Base environment:
    $ arch
    x86_64
    $ docker run --rm -it --platform linux/arm64 --workdir /tmp fedora:40 bash
    $ arch
    aarch64
    # Install uv:
    $ curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.cargo/env
    # Create and enter venv:
    $ uv venv && source .venv/bin/activate
    # Install:
    $ uv pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio
    Resolved 13 packages in 20.18s
    Downloaded 13 packages in 12.26s
    Installed 13 packages in 486ms
     + filelock==3.13.1
     + fsspec==2024.2.0
     + jinja2==3.1.3
     + markupsafe==2.1.5
     + mpmath==1.3.0
     + networkx==3.2.1
     + numpy==1.26.3
     + pillow==10.2.0
     + sympy==1.12
     + torch==2.3.0
     + torchaudio==2.3.0
     + torchvision==0.18.0
     + typing-extensions==4.9.0
    ```
    ---
    You can kind of see why PyTorch has local identifier variants when comparing to PyPi?:
    ```console
    # ARM64 (PyPi instead of PyTorch)
    # Slightly newer versions of the packages but otherwise equivalent
    $ uv pip install torch torchvision torchaudio
    Resolved 13 packages in 1.08s
    Downloaded 13 packages in 23.73s
    Installed 13 packages in 425ms
     + filelock==3.14.0
     + fsspec==2024.3.1
     + jinja2==3.1.4
     + markupsafe==2.1.5
     + mpmath==1.3.0
     + networkx==3.3
     + numpy==1.26.4
     + pillow==10.3.0
     + sympy==1.12
     + torch==2.3.0
     + torchaudio==2.3.0
     + torchvision==0.18.0
     + typing-extensions==4.11.0
    ```
    [CODE]
    ---
    ## Resolve platform packaging inconsistency upstream
    Maybe open an issue about it upstream since it seems to be more of a packaging concern unrelated to `uv`?:
    https://github.com/pytorch/pytorch/issues/110791#issuecomment-1753334468
    > _please do not hesitate to open a new issue (or a PR) if you want to propose dropping `+cpu` suffix for Linux and Windows wheels, as nobody seems to be relying on the old installation method anyway._
    Which makes sense that CPU should be the default, with local identifiers only used for actual variants (_eg: different cuda version support_). Might introduce some friction to any existing CI perhaps, but you can see similar with nvidia's own Docker images they publish where the naming convention isn't always consistent 😅 (_at least it broke a PyTorch docker build CI task recently_)
    
    ==================================================
    
    Query: Can I add a package dependency without version requirement?
    
    Result URL: https://github.com/astral-sh/uv/issues/6476#issuecomment-2305937941
    
    Result:
    Issue 6476: Allow adding a dependency with no version constraint
    
    `uv add` adds a lower bound version constraint by default. For example, calling `uv add requests` currently adds the dependency `"requests>=2.32.3"` to `pyproject.toml`.
    It's possible to have uv use different upper and/or lower bounds, but I could not find a way to add a dependency without any version constraint at all. For example, I want the dependency `"requests"` added to `pyproject.toml` without any bounds.
    The current default behavior is fine, but I think there should be some way to add an unconstrained dependency, either with a global configuration setting that changes the default behavior, or with a command-line option for `uv add`.
    Apologies if I missed something in the documentation and this is already possible. (Of course I could edit `pyproject.toml` manually, but it doesn't seem like that should be necessary.)
    Thanks for the great work on a fantastic project!
    
    Comment:
    That seems ok to me.
    
    ==================================================
    
    Query: What does the `.python-version` file do?
    
    Result URL: https://github.com/astral-sh/uv/issues/8920#issuecomment-2465005344
    
    Result:
    Issue 8920: Purpose of .python-version?
    
    [COMMENT]
    Currently on `uv init` a project boilerplate created includes the file `.python-version`.
    What is the purpose of this file, if there are constraints in `pyproject.toml` and `uv.lock`?
    Is it just for compatibility with tools like pyenv or is there more to it?
    I really like how clean project directory became after moving a lot of stuff (like `requirements.*`) to `pyproject.toml`. So wonder, if `.python-version` is really required in two senses:
    1. Is it required for the projects now? I tested by deleting `.python-version`, and everything works as expected and this file is not recreated with commands like `uv run`, `uv sync` and `uv lock`
    2. Since a lot of constraints/pinned versions of things are now in `pyproject.toml` and `uv.lock`, maybe if `.python-version` still plays some important role, its function can be moved to `pyproject.toml` or `uv.lock` in future?
    Asking as a 5-year user of `pyenv` and `.python-version`. They are great, but I don't really miss them.
    Also, couldn't find peps related to this file. 
    
    Comment:
    Related - https://github.com/astral-sh/uv/issues/8247
    A `.python-version` file is not strictly required, but it's useful to have it when developing a project as it allows you to specify the exact Python version you are using to do development. It's different to the `requires-python` field  - that is the range of Python versions supported by your project.
    
    
    ==================================================
    


[1]: https://huggingface.co/learn/nlp-course/en/chapter5/5
[2]: make-huggingface-dataset-of-github-repo-issues.md
[3]: https://huggingface.co/BAAI/bge-m3
[4]: https://huggingface.co/FacebookAI/xlm-roberta-large
[5]: https://github.com/facebookresearch/faiss
[6]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-below-1m-vectors-ivfk
[7]: https://huggingface.co/BAAI/bge-reranker-v2-m3
