---
date:
  created: 2025-03-22
  updated: 2025-03-29

categories:
- Data preparation

tags:
- Polars
- Hugging Face

slug: clean-uci-drug-review-dataset
---

# Cleaning UCI ML Drug Review Dataset

This is my [:simple-polars: Polars][1]-centric adaptation of the tutorial
[:simple-huggingface: Hugging Face NLP Course: Time to slice and dice][2],
which cleans the UCI ML Drug Review dataset available at
[:fontawesome-brands-kaggle: Kaggle][3].

:material-database: Cleaned dataset: [dd-n-kk/uci-drug-review-cleaned][4]

<a href="https://colab.research.google.com/github/dd-n-kk/notebooks/blob/main/blog/clean-uci-drug-review-dataset.ipynb" target="_parent">
    :simple-googlecolab: Open in Colab
</a>

<!-- more -->

## Preparations


```python
# Set this to an empty string to avoid saving the dataset file.
DIRPATH = "/content/drive/MyDrive/uci-drug-review-cleaned/"

# Set these to empty strings to avoid uploading the dataset.
REPO_ID = "dd-n-kk/uci-drug-review-cleaned"
COLAB_SECRET = "HF_TOKEN"
```


```python
!uv pip install --system -Uq polars
```


```python
import kagglehub
import polars as pl
from polars import col
```


```python
path = kagglehub.dataset_download("jessicali9530/kuc-hackathon-winter-2018")
!mkdir data/ && cp -t data/ {path}/* && ls -hAil data
```

    Downloading from https://www.kaggle.com/api/v1/datasets/download/jessicali9530/kuc-hackathon-winter-2018?dataset_version_number=2...


    100%|██████████| 40.7M/40.7M [00:00<00:00, 44.6MB/s]

    Extracting files...


    


    total 106M
    2364654 -rw-r--r-- 1 root root 27M Mar 22 13:17 drugsComTest_raw.csv
    2364655 -rw-r--r-- 1 root root 80M Mar 22 13:17 drugsComTrain_raw.csv



```python
SEED = 777
pl.set_random_seed(SEED)

# Configure Polars for more complete display.
_ = pl.Config(
    tbl_cols=-1,
    tbl_rows=100,
    tbl_width_chars=-1,
    float_precision=3,
    fmt_str_lengths=500,
    fmt_table_cell_list_len=-1,
)
```


```python
df = pl.read_csv("data/drugsComTrain_raw.csv")
```

## Observations

- `drugName`:
    - Capitalization is inconsistent ("A + D Cracked Skin Relief" vs. "femhrt").
    - There are combination prescriptions ("Ethinyl estradiol / norgestimate").

- `condition`:
    - There are 899 nulls.
    - There are corrupted values:
        - "0`</span>` users found this comment helpful."
        - "zen Shoulde" should probably be "Frozen Shoulder".

- `review`:
    - There are HTML entities (`&#039;`).
    - There are emojis (❤️❤️❤️).

- `date`:
    - The format could be simplified ("1-Apr-08").


```python
df.describe()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (9, 8)</small><table border="1" class="dataframe"><thead><tr><th>statistic</th><th>uniqueID</th><th>drugName</th><th>condition</th><th>review</th><th>rating</th><th>date</th><th>usefulCount</th></tr><tr><td>str</td><td>f64</td><td>str</td><td>str</td><td>str</td><td>f64</td><td>str</td><td>f64</td></tr></thead><tbody><tr><td>&quot;count&quot;</td><td>161297.000</td><td>&quot;161297&quot;</td><td>&quot;160398&quot;</td><td>&quot;161297&quot;</td><td>161297.000</td><td>&quot;161297&quot;</td><td>161297.000</td></tr><tr><td>&quot;null_count&quot;</td><td>0.000</td><td>&quot;0&quot;</td><td>&quot;899&quot;</td><td>&quot;0&quot;</td><td>0.000</td><td>&quot;0&quot;</td><td>0.000</td></tr><tr><td>&quot;mean&quot;</td><td>115923.585</td><td>null</td><td>null</td><td>null</td><td>6.994</td><td>null</td><td>28.005</td></tr><tr><td>&quot;std&quot;</td><td>67004.445</td><td>null</td><td>null</td><td>null</td><td>3.272</td><td>null</td><td>36.404</td></tr><tr><td>&quot;min&quot;</td><td>2.000</td><td>&quot;A + D Cracked Skin Relief&quot;</td><td>&quot;0&lt;/span&gt; users found this comment helpful.&quot;</td><td>&quot;&quot;

 please tell the ones who is suffering from anxiety to use lavender chamomile spray by air wick.&nbsp;&nbsp;it gives immediate relief , doctors not letting know patients about this. please spread the word!!.&nbsp;&nbsp;Please keep this post here.&quot;&quot;</td><td>1.000</td><td>&quot;1-Apr-08&quot;</td><td>0.000</td></tr><tr><td>&quot;25%&quot;</td><td>58063.000</td><td>null</td><td>null</td><td>null</td><td>5.000</td><td>null</td><td>6.000</td></tr><tr><td>&quot;50%&quot;</td><td>115744.000</td><td>null</td><td>null</td><td>null</td><td>8.000</td><td>null</td><td>16.000</td></tr><tr><td>&quot;75%&quot;</td><td>173776.000</td><td>null</td><td>null</td><td>null</td><td>10.000</td><td>null</td><td>36.000</td></tr><tr><td>&quot;max&quot;</td><td>232291.000</td><td>&quot;femhrt&quot;</td><td>&quot;zen Shoulde&quot;</td><td>&quot;&quot;❤️❤️❤️ Cialis for US!!&nbsp;&nbsp;&nbsp;&nbsp;I wish I had my husband start this decades ago &quot;&quot;</td><td>10.000</td><td>&quot;9-Sep-17&quot;</td><td>1291.000</td></tr></tbody></table></div>




```python
df.sample(10, seed=SEED)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 7)</small><table border="1" class="dataframe"><thead><tr><th>uniqueID</th><th>drugName</th><th>condition</th><th>review</th><th>rating</th><th>date</th><th>usefulCount</th></tr><tr><td>i64</td><td>str</td><td>str</td><td>str</td><td>i64</td><td>str</td><td>i64</td></tr></thead><tbody><tr><td>63419</td><td>&quot;Epiduo&quot;</td><td>&quot;Acne&quot;</td><td>&quot;&quot;I have been using epiduo for almost a week now and it has been burning a little bit, so i cut back on how much i put on my face and it feels much better. I did not have alot of acne at all before using it but only a few on my cheeks and those are all gone now after a few days of use. The few bumps on ny forehead are still there but im positive the epiduo is fighting to get rid of it. I will give it a couple of weeks :). (avoid using in eye area, and area right AROUND nose... those are sensitive…</td><td>8</td><td>&quot;27-Jun-16&quot;</td><td>4</td></tr><tr><td>152744</td><td>&quot;Doxycycline&quot;</td><td>&quot;Bacterial Infection&quot;</td><td>&quot;&quot;Have been taking this med now for the past 8 days, for an acute sinusitis infection, and paid strict attention to the pharmacists instructions..... YOU MUST DRINK A FULL GLASS OF WATER after you&amp;#039;ve taken it.&nbsp;&nbsp;Don&amp;#039;t disregard these rules, or you will have severe side effects.&nbsp;&nbsp;&nbsp;Otherwise I&amp;#039;m hoping this time, doxy will clear it up as I was on amoxy and that didn&amp;#039;t. This infection keeps repeating because of two bouts of pneumonia, and I&amp;#039;m sure residual infection wasn&amp;#039…</td><td>8</td><td>&quot;3-Nov-17&quot;</td><td>2</td></tr><tr><td>126485</td><td>&quot;Brimonidine&quot;</td><td>&quot;Rosacea&quot;</td><td>&quot;&quot;Stay away from this medication. The 1st day used, I was impressed, but 2 days later I had a rebound and my face was burning like hell. I waited like 4 days to re-apply and see what happens, but I lost sensation on my lips and they got swollen. Then, after 24 hours the rebound got worse and was like two weeks after my skin stopped burning. I still don&amp;#039;t understand how this product was approved by the FDA.&quot;&quot;</td><td>1</td><td>&quot;27-Jun-16&quot;</td><td>21</td></tr><tr><td>208405</td><td>&quot;Oseltamivir&quot;</td><td>&quot;Influenza&quot;</td><td>&quot;&quot;Sick sick with flu, tamiflu added to illness, nausea &amp;amp; vomiting, don&amp;#039;t know if made flu last shorter, made me too nauseous to know, just ride out the virus, plan on being sick for a week, don&amp;#039;t burden your body with meds that may make you sick&quot;&quot;</td><td>2</td><td>&quot;25-Dec-15&quot;</td><td>17</td></tr><tr><td>166105</td><td>&quot;Levonorgestrel&quot;</td><td>null</td><td>&quot;&quot;Well I&amp;#039;m the first one reviewing this... I got my Kyleena IUD placed on Dec. 2nd 2016. It was painful but no more so than other IUDs I&amp;#039;ve read about. The issues I&amp;#039;m having is contant spotting. I&amp;#039;ve been bleeding lightly (where I have to wear a light pad so I don&amp;#039;t ruin my underwear every day) for two months now. The office told me about a month ago that my spotting is normal and most likely means once I stop spotting I&amp;#039;ll be done with my period for the five years K…</td><td>5</td><td>&quot;30-Jan-17&quot;</td><td>17</td></tr><tr><td>16075</td><td>&quot;Ethinyl estradiol / norethindrone&quot;</td><td>&quot;Birth Control&quot;</td><td>&quot;&quot;I&amp;#039;m 14 years old and I&amp;#039;ve been on it for 4 months and I hate it. I&amp;#039;m on it for regular periods since my blood flows too quickly and it has done nothing. I&amp;#039;ve been on my period for 4 months, light mostly but heavy sometimes. I also went on it for my acne but it did nothing. I gained 4 pounds and now I have stubborn belly fat. I also have huge stretch marks (idk if its related but they showed up when I took it). Also, I have no motivation anymore. Don&amp;#039;t take this please!&quot;&quot;</td><td>3</td><td>&quot;27-May-16&quot;</td><td>4</td></tr><tr><td>223196</td><td>&quot;Estradiol&quot;</td><td>&quot;Postmenopausal Symptoms&quot;</td><td>&quot;&quot;I have had a great experience with the Minivelle patch.&nbsp;&nbsp;I was suffering with many, many, hot flashes a day and night sweats that left me drenched!&nbsp;&nbsp;I am down to a handful of flashes a day and the night sweats are gone!&quot;&quot;</td><td>10</td><td>&quot;15-Jul-13&quot;</td><td>41</td></tr><tr><td>102766</td><td>&quot;Aripiprazole&quot;</td><td>&quot;Major Depressive Disorde&quot;</td><td>&quot;&quot;Weight gain and fatigue.&quot;&quot;</td><td>3</td><td>&quot;29-Jul-14&quot;</td><td>15</td></tr><tr><td>85534</td><td>&quot;Ethinyl estradiol / norgestimate&quot;</td><td>&quot;Birth Control&quot;</td><td>&quot;&quot;This Is An Amazing Birth Control. I&amp;#039;ve Been On This For Five Years And Counting. I Never Got Pregnant, My Menstrual Last About 3-4 Days(Regular Period ) ,No Cramps, Face Is Clear Of Acne, From Time To Time Ive Gotten Moody But Overall I Love It!&nbsp;&nbsp;I&amp;#039;m Looking Forward To Getting Off And Starting A Family.&quot;&quot;</td><td>9</td><td>&quot;17-Feb-15&quot;</td><td>24</td></tr><tr><td>131566</td><td>&quot;Methyl salicylate&quot;</td><td>&quot;Muscle Pain&quot;</td><td>&quot;&quot;The adhesive used for these particular patches works a lot better than the other comparable products on the market by far. As long as you get them on straight and unwrinkled the first time, they will stay on for virtually as long as they last, which also happens to exceed comparable products by far. Couple this with the much less expensive price and you have three reasons why this is simply a superior product to the others. 

It should be noted that the one time I almost bought another name b…</td><td>7</td><td>&quot;7-Jun-10&quot;</td><td>36</td></tr></tbody></table></div>



## Verifying uniqueness of `uniqueID`

Unlike the version used in the Hugging Face tutorial,
the Kaggle-hosted dataset does have a name for the first column.
So here we just verify that each row indeed has a unique ID.


```python
assert df.get_column("uniqueID").n_unique() == len(df)
```

## Scanning for invalid `drugName`s

We scan for invalid drug names by looking for unusual characters.
Fortunately most if not all drug names seem to be valid.


```python
(
    df.filter(col("drugName").str.contains(r"[^[:alnum:] '(),./-]"))
    .get_column("drugName")
    .value_counts()
    .sort("count", descending=True)
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (17, 2)</small><table border="1" class="dataframe"><thead><tr><th>drugName</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;Tylenol with Codeine #3&quot;</td><td>65</td></tr><tr><td>&quot;Coricidin HBP Cold &amp; Flu&quot;</td><td>5</td></tr><tr><td>&quot;Tylenol with Codeine #4&quot;</td><td>4</td></tr><tr><td>&quot;Mucinex Fast-Max Severe Congestion &amp; Cough&quot;</td><td>2</td></tr><tr><td>&quot;Dimetapp Children&#x27;s Cold &amp; Cough&quot;</td><td>2</td></tr><tr><td>&quot;Aleve-D Sinus &amp; Cold&quot;</td><td>1</td></tr><tr><td>&quot;Coricidin HBP Cough &amp; Cold&quot;</td><td>1</td></tr><tr><td>&quot;Sudafed PE Pressure + Pain&quot;</td><td>1</td></tr><tr><td>&quot;A + D Cracked Skin Relief&quot;</td><td>1</td></tr><tr><td>&quot;Norinyl 1+35&quot;</td><td>1</td></tr><tr><td>&quot;PanOxyl 10% Acne Foaming Wash&quot;</td><td>1</td></tr><tr><td>&quot;Hair, Skin &amp; Nails&quot;</td><td>1</td></tr><tr><td>&quot;Vicks Dayquil Cold &amp; Flu Relief&quot;</td><td>1</td></tr><tr><td>&quot;Robitussin Cough + Chest Congestion DM&quot;</td><td>1</td></tr><tr><td>&quot;Excedrin Back &amp; Body&quot;</td><td>1</td></tr><tr><td>&quot;Mucinex Fast-Max Night Time Cold &amp; Flu&quot;</td><td>1</td></tr><tr><td>&quot;Vitafol-OB+DHA&quot;</td><td>1</td></tr></tbody></table></div>



## Handling capitalization of `drugName`s

Fortunately there are very few lowercase drug names and all of them are valid.
We can also confirm that no drug name is cased differently in the dataset.
To preserve the original data as much as possible,
I title-case the lowercase names instead of lowercasing all drug names.


```python
(
    df.filter(col("drugName").str.contains(r"^[^A-Z]"))
    .get_column("drugName")
    .value_counts()
    .sort("count", descending=True)
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (3, 2)</small><table border="1" class="dataframe"><thead><tr><th>drugName</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;ella&quot;</td><td>51</td></tr><tr><td>&quot;femhrt&quot;</td><td>3</td></tr><tr><td>&quot;depo-subQ provera 104&quot;</td><td>1</td></tr></tbody></table></div>




```python
assert (
    df.get_column("drugName").n_unique()
    == df.get_column("drugName").str.to_lowercase().n_unique()
)
```


```python
replacements = {
    "ella": "Ella",
    "femhrt": "Femhrt",
    "depo-subQ provera 104": "Depo-SubQ Provera 104"
}

df = df.with_columns(col("drugName").replace(replacements))
```

## Cleaning invalid `condition`s

The search for unusual characters yields a group of rows with invalid `condition`s:
"??`</span>` users found this comment helpful." I decide to replace them with nulls.


```python
q = df.filter(col("condition").str.contains(r"[^[:alnum:] '(),./-]")); len(q)
```




    900




```python
q.select(pl.exclude("review")).sample(10, seed=SEED)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 6)</small><table border="1" class="dataframe"><thead><tr><th>uniqueID</th><th>drugName</th><th>condition</th><th>rating</th><th>date</th><th>usefulCount</th></tr><tr><td>i64</td><td>str</td><td>str</td><td>i64</td><td>str</td><td>i64</td></tr></thead><tbody><tr><td>81207</td><td>&quot;Yaz&quot;</td><td>&quot;0&lt;/span&gt; users found this comment helpful.&quot;</td><td>1</td><td>&quot;11-Feb-17&quot;</td><td>0</td></tr><tr><td>126293</td><td>&quot;Viibryd&quot;</td><td>&quot;15&lt;/span&gt; users found this comment helpful.&quot;</td><td>5</td><td>&quot;14-Oct-11&quot;</td><td>15</td></tr><tr><td>24826</td><td>&quot;Deplin&quot;</td><td>&quot;13&lt;/span&gt; users found this comment helpful.&quot;</td><td>6</td><td>&quot;9-Aug-16&quot;</td><td>13</td></tr><tr><td>67383</td><td>&quot;Provera&quot;</td><td>&quot;4&lt;/span&gt; users found this comment helpful.&quot;</td><td>1</td><td>&quot;27-Mar-16&quot;</td><td>4</td></tr><tr><td>212726</td><td>&quot;Phenergan&quot;</td><td>&quot;3&lt;/span&gt; users found this comment helpful.&quot;</td><td>10</td><td>&quot;3-Jan-09&quot;</td><td>3</td></tr><tr><td>220738</td><td>&quot;Loestrin 24 Fe&quot;</td><td>&quot;9&lt;/span&gt; users found this comment helpful.&quot;</td><td>5</td><td>&quot;13-Aug-10&quot;</td><td>9</td></tr><tr><td>33084</td><td>&quot;Seasonique&quot;</td><td>&quot;2&lt;/span&gt; users found this comment helpful.&quot;</td><td>8</td><td>&quot;29-Jun-10&quot;</td><td>2</td></tr><tr><td>159396</td><td>&quot;Estrace Vaginal Cream&quot;</td><td>&quot;64&lt;/span&gt; users found this comment helpful.&quot;</td><td>8</td><td>&quot;14-Sep-11&quot;</td><td>64</td></tr><tr><td>209258</td><td>&quot;Geodon&quot;</td><td>&quot;5&lt;/span&gt; users found this comment helpful.&quot;</td><td>6</td><td>&quot;18-Feb-09&quot;</td><td>5</td></tr><tr><td>71470</td><td>&quot;Levora&quot;</td><td>&quot;2&lt;/span&gt; users found this comment helpful.&quot;</td><td>3</td><td>&quot;15-Sep-11&quot;</td><td>2</td></tr></tbody></table></div>




```python
df = df.with_columns(
    pl.when(col("condition").str.contains("</span>", literal=True))
    .then(pl.lit(None))
    .otherwise(col("condition"))
    .alias("condition")
)
```

## Correcting corrupted `condition`s

The search for lowercase condition names leads to a few unexpected findings:

- Some `condition` fields mistakenly record drug names instead of condition names.
- Quite a few condition names lost their prefixes, suffixes, or both.
  Indeed, seemingly all leading `F`s and terminating `r`s were cut off.

I decide to handle the corrupted entries one by one:
Correct those I can recognize and nullify those I cannot.


```python
(
    df.filter(col("condition").str.contains(r"^[^A-Z]"))
    .get_column("condition")
    .value_counts()
    .sort("count", descending=True)
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (37, 2)</small><table border="1" class="dataframe"><thead><tr><th>condition</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;ibromyalgia&quot;</td><td>1791</td></tr><tr><td>&quot;mance Anxiety&quot;</td><td>187</td></tr><tr><td>&quot;moterol)&quot;</td><td>79</td></tr><tr><td>&quot;emale Infertility&quot;</td><td>65</td></tr><tr><td>&quot;atigue&quot;</td><td>53</td></tr><tr><td>&quot;min)&quot;</td><td>45</td></tr><tr><td>&quot;ge (amlodipine / valsartan)&quot;</td><td>33</td></tr><tr><td>&quot;acial Wrinkles&quot;</td><td>32</td></tr><tr><td>&quot;moterol / mometasone)&quot;</td><td>29</td></tr><tr><td>&quot;min / sitagliptin)&quot;</td><td>26</td></tr><tr><td>&quot;zen Shoulde&quot;</td><td>14</td></tr><tr><td>&quot;eve&quot;</td><td>12</td></tr><tr><td>&quot;mulation) (phenylephrine)&quot;</td><td>12</td></tr><tr><td>&quot;min / saxagliptin)&quot;</td><td>10</td></tr><tr><td>&quot;lic Acid Deficiency&quot;</td><td>8</td></tr><tr><td>&quot;von Willebrand&#x27;s Disease&quot;</td><td>7</td></tr><tr><td>&quot;amilial Mediterranean Feve&quot;</td><td>6</td></tr><tr><td>&quot;min / pioglitazone)&quot;</td><td>5</td></tr><tr><td>&quot;mis&quot;</td><td>5</td></tr><tr><td>&quot;ge HCT (amlodipine / hydrochlorothiazide / valsartan)&quot;</td><td>4</td></tr><tr><td>&quot;cal Segmental Glomerulosclerosis&quot;</td><td>4</td></tr><tr><td>&quot;ailure to Thrive&quot;</td><td>3</td></tr><tr><td>&quot;amilial Cold Autoinflammatory Syndrome&quot;</td><td>3</td></tr><tr><td>&quot;actor IX Deficiency&quot;</td><td>3</td></tr><tr><td>&quot;t Pac with Cyclobenzaprine (cyclobenzaprine)&quot;</td><td>2</td></tr><tr><td>&quot;t Care&quot;</td><td>2</td></tr><tr><td>&quot;min / rosiglitazone)&quot;</td><td>2</td></tr><tr><td>&quot;tic (mycophenolic acid)&quot;</td><td>2</td></tr><tr><td>&quot;llicular Lymphoma&quot;</td><td>1</td></tr><tr><td>&quot;ungal Pneumonia&quot;</td><td>1</td></tr><tr><td>&quot;mist (&quot;</td><td>1</td></tr><tr><td>&quot;m Pain Disorde&quot;</td><td>1</td></tr><tr><td>&quot;me&quot;</td><td>1</td></tr><tr><td>&quot;unctional Gastric Disorde&quot;</td><td>1</td></tr><tr><td>&quot;acial Lipoatrophy&quot;</td><td>1</td></tr><tr><td>&quot;ibrocystic Breast Disease&quot;</td><td>1</td></tr><tr><td>&quot;ungal Infection Prophylaxis&quot;</td><td>1</td></tr></tbody></table></div>




```python
replacements = {
    "ibromyalgia": "Fibromyalgia",
    "mance Anxiety": "Performance Anxiety",
    "emale Infertility": "Female Infertility",
    "atigue": "Fatigue",
    "acial Wrinkles": "Facial Wrinkles",
    "zen Shoulde": "Frozen Shoulder",
    "eve": "Fever",
    "lic Acid Deficiency": "Folic Acid Deficiency",
    "von Willebrand's Disease": "Von Willebrand's Disease",
    "amilial Mediterranean Feve": "Familial Mediterranean Fever",
    "cal Segmental Glomerulosclerosis": "Focal Segmental Glomerulosclerosis",
    "ailure to Thrive": "Failure to Thrive",
    "amilial Cold Autoinflammatory Syndrome": "Familial Cold Autoinflammatory Syndrome",
    "actor IX Deficiency": "Factor IX Deficiency",
    "acial Lipoatrophy": "Facial Lipoatrophy",
    "ungal Pneumonia": "Fungal Pneumonia",
    "llicular Lymphoma": "Follicular Lymphoma",
    "unctional Gastric Disorde": "Functional Gastric Disorder",
    "ungal Infection Prophylaxis": "Fungal Infection Prophylaxis",
    "ibrocystic Breast Disease": "Fibrocystic Breast Disease",
    "llicle Stimulation": "Follicle Stimulation",

    "moterol)": None,
    "min)": None,
    "ge (amlodipine / valsartan)": None,
    "moterol / mometasone)": None,
    "min / sitagliptin)": None,
    "mulation) (phenylephrine)": None,
    "min / saxagliptin)": None,
    "mis": None,
    "min / pioglitazone)": None,
    "ge HCT (amlodipine / hydrochlorothiazide / valsartan)": None,
    "min / rosiglitazone)": None,
    "tic (mycophenolic acid)": None,
    "t Care": None,
    "t Pac with Cyclobenzaprine (cyclobenzaprine)": None,
    "me": None,
    "mist (": None,
    "m Pain Disorde": None,
}

df = df.with_columns(col("condition").replace(replacements))
```

Identifying all conditions losing their terminating `r`s would require nontrivial
domain knowledge and effort.
I can only come up with and correct "disorde" and "(f)eve",
which should be "disorder" and "fever", respectively.


```python
(
    df.filter(col("condition").str.ends_with("eve"))
    .get_column("condition")
    .value_counts()
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (3, 2)</small><table border="1" class="dataframe"><thead><tr><th>condition</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;Q Feve&quot;</td><td>1</td></tr><tr><td>&quot;Typhoid Feve&quot;</td><td>2</td></tr><tr><td>&quot;Pain/Feve&quot;</td><td>13</td></tr></tbody></table></div>




```python
(
    df.filter(col("condition").str.ends_with("rde"))
    .get_column("condition")
    .value_counts()
)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (27, 2)</small><table border="1" class="dataframe"><thead><tr><th>condition</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;Binge Eating Disorde&quot;</td><td>72</td></tr><tr><td>&quot;Temporomandibular Joint Disorde&quot;</td><td>30</td></tr><tr><td>&quot;Bleeding Disorde&quot;</td><td>7</td></tr><tr><td>&quot;Social Anxiety Disorde&quot;</td><td>389</td></tr><tr><td>&quot;Borderline Personality Disorde&quot;</td><td>151</td></tr><tr><td>&quot;Hypoactive Sexual Desire Disorde&quot;</td><td>7</td></tr><tr><td>&quot;Shift Work Sleep Disorde&quot;</td><td>60</td></tr><tr><td>&quot;Post Traumatic Stress Disorde&quot;</td><td>314</td></tr><tr><td>&quot;Somatoform Pain Disorde&quot;</td><td>1</td></tr><tr><td>&quot;Dissociative Identity Disorde&quot;</td><td>1</td></tr><tr><td>&quot;Bipolar Disorde&quot;</td><td>4224</td></tr><tr><td>&quot;Tic Disorde&quot;</td><td>2</td></tr><tr><td>&quot;Persistent Depressive Disorde&quot;</td><td>12</td></tr><tr><td>&quot;Major Depressive Disorde&quot;</td><td>1607</td></tr><tr><td>&quot;Schizoaffective Disorde&quot;</td><td>396</td></tr><tr><td>&quot;Paranoid Disorde&quot;</td><td>38</td></tr><tr><td>&quot;Cyclothymic Disorde&quot;</td><td>5</td></tr><tr><td>&quot;Auditory Processing Disorde&quot;</td><td>5</td></tr><tr><td>&quot;Seasonal Affective Disorde&quot;</td><td>16</td></tr><tr><td>&quot;Periodic Limb Movement Disorde&quot;</td><td>43</td></tr><tr><td>&quot;Panic Disorde&quot;</td><td>1463</td></tr><tr><td>&quot;Intermittent Explosive Disorde&quot;</td><td>2</td></tr><tr><td>&quot;Generalized Anxiety Disorde&quot;</td><td>1164</td></tr><tr><td>&quot;Oppositional Defiant Disorde&quot;</td><td>1</td></tr><tr><td>&quot;Obsessive Compulsive Disorde&quot;</td><td>579</td></tr><tr><td>&quot;Body Dysmorphic Disorde&quot;</td><td>9</td></tr><tr><td>&quot;Premenstrual Dysphoric Disorde&quot;</td><td>298</td></tr></tbody></table></div>




```python
df = df.with_columns(
    col("condition").str.replace(r"^(.+(?:rd|ev)e)$", "${1}r").alias("condition")
)
```

## Decoding HTML entities in `review`s

It turns out that many HTML entities besides `&#039;` are present in the `review` column
and using `html.unescape()` as recommended by the Hugging Face tutorial
is probably more robust than replacing them individually. So we do just that.


```python
q = (
    df.select(col("review").str.extract_all(r"&#?[[:alnum:]]+;"))
    .explode("review")
    .get_column("review")
    .value_counts()
    .sort("count", descending=True)
); len(q)
```




    45




```python
q.head(10)
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10, 2)</small><table border="1" class="dataframe"><thead><tr><th>review</th><th>count</th></tr><tr><td>str</td><td>u32</td></tr></thead><tbody><tr><td>&quot;&amp;#039;&quot;</td><td>262415</td></tr><tr><td>null</td><td>55984</td></tr><tr><td>&quot;&amp;quot;&quot;</td><td>21262</td></tr><tr><td>&quot;&amp;amp;&quot;</td><td>13047</td></tr><tr><td>&quot;&amp;rsquo;&quot;</td><td>3029</td></tr><tr><td>&quot;&amp;gt;&quot;</td><td>162</td></tr><tr><td>&quot;&amp;rdquo;&quot;</td><td>117</td></tr><tr><td>&quot;&amp;ldquo;&quot;</td><td>116</td></tr><tr><td>&quot;&amp;eacute;&quot;</td><td>111</td></tr><tr><td>&quot;&amp;lt;&quot;</td><td>108</td></tr></tbody></table></div>




```python
import html

df = df.with_columns(col("review").map_elements(html.unescape, return_dtype=pl.String))
```

## Simplifying newlines in `review`s

Another issue of the `review`s pointed out by the Hugging Face tutorial is
the presence of many different kinds of newline characters.
I decide to replace all consecutive newline characters with one `\n`.


```python
df.filter(col("uniqueID") == 121260).item(0, "review")
```




    '"More than Depression I take Effexor for Anxiety and it works well.  The only problem\r\r\noccurs when I forget to take it.  Within in hours I experience withdrawal symptoms\r\r\nsuch as light headiness and an occasional brief buzzing in my head."'




```python
df = df.with_columns(col("review").str.replace_all(r"[\r\n]+", "\n"))
```


```python
df.filter(col("uniqueID") == 121260).item(0, "review")
```




    '"More than Depression I take Effexor for Anxiety and it works well.  The only problem\noccurs when I forget to take it.  Within in hours I experience withdrawal symptoms\nsuch as light headiness and an occasional brief buzzing in my head."'



## Reformatting `date`

Finally, I reformat the `dates` into "yyyy-mm-dd", which is sortable as string.


```python
df = df.with_columns(col("date").str.to_date("%-d-%b-%y"))
```

## Saving and sharing


```python
if DIRPATH:
    import os
    from google.colab import drive

    drive.mount('/content/drive')
    os.makedirs(DIRPATH, exist_ok=True)
    df.write_csv(f"{DIRPATH}/train.tsv", separator="\t")
```

    Mounted at /content/drive


The test set is processed the same way and the procedure is omitted for brevity.


```python
if REPO_ID and COLAB_SECRET:
    !uv pip install --system -q datasets

    from datasets import load_dataset
    from google.colab import userdata

    dataset = load_dataset(
        "csv",
        data_files=dict(train=f"{DIRPATH}/train.tsv", test=f"{DIRPATH}/test.tsv"),
        delimiter="\t",
    )

    dataset.push_to_hub(REPO_ID, token=userdata.get(COLAB_SECRET))
```


    Uploading the dataset shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Creating parquet from Arrow format:   0%|          | 0/162 [00:00<?, ?ba/s]



    Uploading the dataset shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Creating parquet from Arrow format:   0%|          | 0/54 [00:00<?, ?ba/s]



```python
from google.colab import drive
drive.flush_and_unmount()
```

[1]: https://pola.rs/
[2]: https://huggingface.co/learn/nlp-course/chapter5/3
[3]: https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
[4]: https://huggingface.co/datasets/dd-n-kk/uci-drug-review-cleaned
