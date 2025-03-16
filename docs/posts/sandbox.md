---
date:
  created: 2025-03-17
  updated: 2025-03-17
draft: true
categories:
- Miscellany
---

# Sandbox

<!-- more -->


## Python Markdown

- Caret:
    - ^Superscript^
    - ^^Underline^^

- Tilde:
    - ~Subscript~
    - ~~Crossed-out~~

- Mark:
    - ==Marked==

- Footnoted[^1]
  [^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.


## KaTeX

- $\displaystyle \sum_{k=1}^n k^2 = \frac{1}{6}n(n + 1)(2n + 1)$

- $$\frac{1}{σ\sqrt{2π}}\exp\left( -\frac{1}{2}\left(\frac{x-μ}{σ}\right)^2 \right)$$


## Code

- Inline: `#!py lambda seq: {str(x): x for x in seq}`

- Block:
  ```py
  def foo(x: int | None = None) -> int:
    return x or 0  # Comment
  ```
