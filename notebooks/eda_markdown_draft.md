# 01_eda.ipynb Markdown 草稿

这是临时文件,不放进 notebook。把下面每段文字复制进对应位置的 markdown cell 就行。用完可以删这个文件。

---

## 位置 1:cell 0 `## Import Library` 下面加一句(可选)

```markdown
载入分析要用的库:pandas 处理数据、matplotlib 画图、statsmodels 做季节分解和平稳性检验。
```

---

## 位置 2:cell 2 `## Load Dataset and Check the Shape of the Dataset` 下面加

```markdown
读取 `src/cleaning.py` 产出的季度面板数据 (`panel.parquet`),先看整体形状、字段、地区种类、时间范围,
以及各地区在 `median_price`、`total_value`、`mean_price`、`rate` 四个字段上的缺失情况,确认数据能不能直接拿来分析。
```

---

## 位置 3:cell 10 `## Median Price By Region` 下面加

```markdown
把 15 个地区的 `median_price` 画在同一张图上,先直观看整体走势:是否都是长期上涨、
涨幅差异大不大、有没有明显的转折点(比如 2020 年后的走势)。
```

---

## 位置 4:cell 12 `## Decompose` 下面加(这段放在 5 个城市小标题之前,统一说明)

```markdown
对 Brisbane、Sydney、Canberra、Adelaide、Melbourne 这五个主要城市做季节性分解 (`seasonal_decompose`,
加法模型,period=4 代表一年四季度),拆出 trend(长期趋势)、seasonal(季节规律)、residual(剩余噪声)
三部分,看价格波动主要来自趋势还是季节性。
```

---

## 位置 5:cell 28 `## Correlation vs Cash Rate` 下面加

```markdown
观察各城市房价与 RBA 现金利率 (`rate`) 的关系,用散点图先直观看,再分别计算「原始价格水平」
和「价格环比增长率」与利率的相关系数,比较两者结果是否一致。
```

---

## 位置 6:cell 38(Adelaide 散点图)和 cell 39(相关系数计算)之间插入新 markdown cell

```markdown
### 相关系数计算

先算「原始价格水平」与利率的相关系数 —— 但两者都是长期趋势序列,这种相关系数可能只是
「两条都在涨」造成的假象(spurious correlation),不代表利率真的影响房价。

所以再算一次「价格环比增长率」(`price_pct_change`)与利率的相关系数,这个更能反映真实关系,
因为增长率已经去掉了长期趋势的干扰。
```

---

## 位置 7:cell 40 和 cell 41 之间插入新 markdown cell(新增小节标题)

```markdown
## 平稳性检验 (ADF Test)

上面比较相关系数时提到「趋势序列容易产生假相关」的问题,这里正式检验:15 个地区的
`median_price` 原始序列是否平稳 (stationary)。用 Augmented Dickey-Fuller (ADF) 检验,
p-value < 0.05 视为平稳。

结果:15 个地区全部不平稳 —— 符合预期,因为房价数据长期都在上涨,不是围绕固定均值波动。
```

---

## 位置 8:cell 41 和 cell 42 之间插入新 markdown cell

```markdown
### 一阶差分后再测一次

原始价格不平稳,接下来做一阶差分 (`.diff()`,即「这一期减上一期」),把长期趋势去掉,
再对差分后的序列重新做 ADF 检验。

结果:15 个地区中 7 个变平稳 (Canberra、Darwin、Melbourne、Rest of NSW、Rest of NT、
Rest of Vic.、Sydney),其余 8 个 (Adelaide、Brisbane、Hobart、Perth、Rest of Qld.、
Rest of SA、Rest of Tas.、Rest of WA) 差分后仍不平稳,可能是波动幅度会随价格水平变大
(方差不稳定),单纯差分处理不了这种情况。
```

---

## 位置 9:cell 42 后面加(整个 notebook 最后,加一个总结)

```markdown
## 小结

- 15 个地区房价长期都在上涨,原始水平数据不平稳。
- 一阶差分后,7 个地区转为平稳,其余 8 个地区差分后仍不平稳,推测是波动幅度随价格水平
  变大导致(方差不稳定),而非单纯的趋势问题。
- 改用 log(median_price) 再差分 (log-diff,近似「环比增长率」) 重新检验:Adelaide、
  Brisbane、Rest of Qld. 转为平稳,确认这三个地区的问题确实出在方差随价格变大,而非趋势。
- 剩下 5 个地区 (Hobart、Perth、Rest of SA、Rest of Tas.、Rest of WA) log-diff 后仍不平稳,
  可能是结构性突变(如疫情冲击、区域性政策)或数据量偏少造成,趋势/方差转换都解决不了。
- 后续建模若用 RF/XGBoost,不强制要求平稳性,但用 log-diff 或价格增长率作为特征,
  比直接用原始价格水平更合理,也能避免前面提到的假相关问题。
- 下一步:定义预测目标(价格水平 vs 增长率)、做特征工程、切分训练/测试集(按时间切,不能随机打乱)、
  训练 baseline 模型。
```

---

## 操作说明

1. 上面每段 ```markdown fenced block``` 里的内容,复制贴到对应位置的 markdown cell。
2. 位置 6、7、8 目前 notebook 里没有对应的 markdown cell,需要在指定两个 cell 中间手动插入新 cell (Jupyter 里选中 cell 按 `b` 插入下方 cell,改成 Markdown 类型)。
3. 全部贴完确认没问题后,把这个 `eda_markdown_draft.md` 删掉。
