---
date:
  created: 2025-03-26
  updated: 2025-03-29

categories:
- Miscellany

tags:
- REST API

slug: get-github-repo-issues-comments
---

# Getting GitHub repo issues and comments

This is Part I of my adaptation of the tutorial
[:simple-huggingface: Hugging Face NLP Course: Creating your own dataset][2].
Other posts in the series:

- Part II: [Making a Hugging Face dataset of GitHub repo issues][1]

<a href="https://colab.research.google.com/github/dd-n-kk/notebooks/blob/main/blog/get-github-repo-issues-comments.ipynb" target="_parent">
    :simple-googlecolab: Open in Colab
</a>

<!-- more -->

## Preparations

We use the [:simple-uv: astral-sh/uv][6] repository as an example.


```python
from google.colab import userdata

SECRET = "GH_TOKEN"
REPO = "astral-sh/uv"
```

## Getting paginated data from GitHub REST API

The Hugging Face tutorial gets comments [issue by issue][3], which is certainly viable
but costs as many requests as the number of issues in the repository,
which often surpasses even an authenticated quota of [5000][5] requests per hour.

Alternatively, we can use the paginated [repository issue comments endpoint][4]
to get up to 100 comments per request.
This also allows getting issues and comments data in a unified approach:


```python
import re
import sys
from datetime import datetime
from typing import Any, Literal

import requests
from requests import Response
from tqdm.auto import tqdm


def get_paginated(
    route: str,
    start_page: int = 1,
    end_page: int | None = None,  # None: Make requests until the last page.
    *,
    token: str | None = None,
    **query: Any,
) -> tuple[list[dict[str, Any]], Response]:
    if start_page < 1 or (end_page is not None and end_page < start_page):
        raise ValueError(f"Invalid page range: [{start_page}, {end_page}]")

    headers = {"Authorization": f"token {token}"} if token else None
    query |= {"page": start_page}
    q = "&".join(f"{k}={v}" for k, v in query.items() if v is not None)

    results = []
    with requests.Session() as session:
        response = session.get(f"https://api.github.com{route}?{q}", headers=headers)
        if response.status_code != 200:
            print(f"{response.status_code}: {response.reason}", file=sys.stderr)
            return results, response

        results.extend(response.json())
        links = parse_link_header(response)
        end_page = _revise_end_page(start_page, end_page, response, links)
        if end_page <= start_page:
            return results, response

        with tqdm(total=end_page - start_page + 1) as progress:
            progress.update(1)
            while "next" in links:
                response = session.get(links["next"], headers=headers)
                if response.status_code != 200:
                    print(f"{response.status_code}: {response.reason}", file=sys.stderr)
                    break

                results.extend(response.json())
                links = parse_link_header(response)
                progress.update(1)

    return results, response


def parse_link_header(response: Response) -> dict[str, str]:
    links = requests.utils.parse_header_links(response.headers["Link"])
    return {link["rel"]: link["url"] for link in links}


def _revise_end_page(
    start_page: int, end_page: int | None, response: Response, links: dict[str, str]
) -> int:
    if not ("last" in links and (m := re.search(r"(?:[?&]page=(\d+))", links["last"]))):
        return start_page

    last = int(m.group(1))
    limit = start_page + int(response.headers["X-RateLimit-Remaining"])

    return min(last, limit) if end_page is None else min(last, limit, end_page)
```


```python
def show_rate_limit(response: Response) -> None:
    h = response.headers
    reset_time = datetime.fromtimestamp(int(h["X-RateLimit-Reset"]))
    now = datetime.now()
    if reset_time > now:
        print(
            f"Requests remaining: {h['X-RateLimit-Remaining']}/{h['X-RateLimit-Limit']}"
            f"\nTime until reset: {str(reset_time - now).split('.')[0]}",
        )
    else:
        print("The rate limit has been reset.")
```

## Caveats and workarounds



Unfortunately, GitHub seems to impose a 300-page limit on paginated query results.
To get more than 30000 issue comments, a workaround is to query for time-sorted entries
and restrict the query range with the `since` parameter.
However, because `since` is based on the last update time,
we should set `sort` and `direction` accordingly to best avoid overlaps between queries.

The wrapper interfaces and their usage are as follows:


```python
# https://docs.github.com/en/rest/issues/issues#list-repository-issues
def get_issues(
    repo: str,
    start_page: int = 1,
    end_page: int | None = None,
    *,
    token: str | None = None,
    per_page: int = 100,
    state: Literal["all", "open", "closed"] = "all",
    sort: Literal["created", "updated", "comments"] = "created",
    direction: Literal["asc", "desc"] = "desc",
    since: str | None = None,  # ⟨YYYY⟩-⟨MM⟩-⟨DD⟩T⟨hh⟩-⟨mm⟩-⟨ss⟩Z
    **query: Any,
) -> tuple[list[dict[str, Any]], Response]:
    return get_paginated(
        f"/repos/{repo}/issues",
        start_page=start_page,
        end_page=end_page,
        token=token,
        per_page=per_page,
        state=state,
        sort=sort,
        direction=direction,
        since=since,
        **query,
    )


# https://docs.github.com/en/rest/issues/comments#list-issue-comments-for-a-repository
def get_issue_comments(
    repo: str,
    start_page: int = 1,
    end_page: int | None = None,
    *,
    token: str | None = None,
    per_page: int = 100,
    sort: Literal["created", "updated"] = "created",
    direction: Literal["asc", "desc"] = "desc",
    since: str | None = None,  # ⟨YYYY⟩-⟨MM⟩-⟨DD⟩T⟨hh⟩-⟨mm⟩-⟨ss⟩Z
    **query: Any,
) -> tuple[list[dict[str, Any]], Response]:
    return get_paginated(
        f"/repos/{repo}/issues/comments",
        start_page=start_page,
        end_page=end_page,
        token=token,
        per_page=per_page,
        sort=sort,
        direction=direction,
        since=since,
        **query,
    )
```


```python
issues, response = get_issues(
    REPO,
    token=userdata.get(SECRET) if SECRET else None,
    state="all",
    sort="updated",
    direction="asc",
)
```


      0%|          | 0/125 [00:00<?, ?it/s]



```python
show_rate_limit(response)
```

    Requests remaining: 4875/5000
    Time until reset: 0:58:30



```python
comments, response = get_issue_comments(
    REPO,
    token=userdata.get(SECRET) if SECRET else None,
    sort="updated",
    direction="asc",
)
```


      0%|          | 0/300 [00:00<?, ?it/s]



```python
comments[-1]["updated_at"]
```




    '2025-02-06T23:52:33Z'




```python
comments_2, response = get_issue_comments(
    REPO,
    token=userdata.get(SECRET) if SECRET else None,
    sort="updated",
    direction="asc",
    since="2025-02-06T23:52:34Z",
)
```


      0%|          | 0/40 [00:00<?, ?it/s]



```python
comments.extend(comments_2)
```


```python
show_rate_limit(response)
```

    Requests remaining: 4535/5000
    Time until reset: 0:49:19


## Saving query results as JSON Lines


```python
import json
from collections.abc import Iterable
from typing import Any


def load_jsonl(path: str) -> list[Any]:
    with open(path, "r") as file:
        return [json.loads(line) for line in file]


def save_jsonl(objs: Iterable[Any], path: str, mode: str = "a") -> None:
    with open(path, mode=mode) as file:
        for obj in objs:
            file.write(json.dumps(obj) + "\n")
```


```python
save_jsonl(issues, "issues.jsonl")
save_jsonl(comments, "comments.jsonl")
```


```python
assert issues == load_jsonl("issues.jsonl")
assert comments == load_jsonl("comments.jsonl")
```

[1]: make-huggingface-dataset-of-github-repo-issues.md
[2]: https://huggingface.co/learn/nlp-course/en/chapter5/5
[3]: https://docs.github.com/en/rest/issues/comments?#list-issue-comments
[4]: https://docs.github.com/en/rest/issues/comments?#list-issue-comments-for-a-repository
[5]: https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?#primary-rate-limit-for-authenticated-users
[6]: https://github.com/astral-sh/uv
