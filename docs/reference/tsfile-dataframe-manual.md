<!-- 本文件由 scripts/sync-feishu-docs.py 自动生成，请勿手工编辑。 -->
<!-- 内容更新：修改飞书原文后重新运行同步脚本。 -->

> **来源**：[飞书文档](https://apache-iotdb-project.feishu.cn/docx/SenJdxlbuoUS5Uxmq7jcOUzdnob)（docx token `SenJdxlbuoUS5Uxmq7jcOUzdnob`）  
> **最后同步**：2026-05-25  
> **更新方式**：`python3 scripts/sync-feishu-docs.py`

---

# TsFileDataFrame用户手册

# 快速上手

TsFileDataFrame能够让你像操作 DataFrame 一样读取TsFile 中的时序数据，无需关心底层文件格式和数据加载细节。

## 表模型示例

```Python
from tsfile import TsFileDataFrame

df = TsFileDataFrame("table_data/")                # 加载目录下所有 TsFile                                   # 浏览所有序列

ts = df["weather.Beijing.humidity"]           # 取一条序列
window = ts[20:100]                           # 按行号切片 -> np.ndarray

data = df.loc[start:end, [                    # 按时间戳对齐多条序列
    "weather.Beijing.temperature",
    "weather.Beijing.humidity",
]]
data.values                                   # -> np.ndarray, shape=(N, 2)
```

## 树模型示例

```Python
from tsfile import TsFileDataFrame

with TsFileDataFrame("tree_data/") as df:        # 加载目录下所有 .tsfile（自动关闭）
    print(df)                                    # 浏览所有序列
    df.list_timeseries("root.db.d1")             # 按路径前缀筛序列名

    ts = df["root.db.d1.s1"]                     # 取一条序列（懒加载）
    window = ts[20:100]                          # 按行号切片 → np.ndarray

    start, end = 1_700_000_000_000, 1_700_003_600_000   # 毫秒时间戳
    data = df.loc[start:end, [                   # 按时间戳对齐多条序列
        "root.db.d1.s1",
        "root.db.d1.s2",
    ]]
    data.values                                  # → np.ndarray, shape=(N, 2)
```

# 接口总览

TsFileDataFrame围绕着三个核心类型：

- **TsFileDataFrame:**入口对象，加载一至多个 TsFile 并提供统一视图。初始化时只扫描元数据，不读取实际数值。
- **Timeseries:**单条时间序列的懒加载句柄。通过 df[...]的数组操作获得，包含序列元信息但不立即读取，仅在行号索引时才触发数据读取
- **AlignedTimeseries:** 多条序列的时间对齐结果。通过df.loc[...]获取，一次性将指定时间范围内的多条序列对齐到同一时间轴并读入内存

## TsFileDataFrame

|  |  |  |
| --- | --- | --- |
| 示例 | 操作 | 返回类型 |
| `TsFileDataFrame(paths)` | 加载文件/目录 | TsFileDataFrame |
| `len(df)` | 获取时间序列总数 | int |
| `df.list_timeseries("weather")` | 获取/按前缀筛选序列名 | List[str] |
| `df.list_timeseries_metadata("weather")` | 获取/按前缀筛选得到序列的元数据信息 | pandas.DataFrame |
| `df["weather.beijing.humidity"]，df[0], df[-1]` | 获取单条序列 | Timeseries |
| `df[0:3], df[[0,2,5]]` | 获取多条序列 | List[Timeseries] |
| `df.loc[start:end, serlies_list]` | 按时间戳对齐查询 | AlignedTimeseries |

## Timeseries

|  |  |  |
| --- | --- | --- |
| 示例 | 操作 | 返回类型 |
| `ts.name` | 序列名 | str |
| `len(ts)` | 序列点数 | int |
| `ts.stats` | 序列统计信息 | dict |
| `ts[20]` | 单值读取 | float |
| `ts[20:100]` | 行范围切片 | np.ndarray |
| `ts.``timestamps` | 时间戳数组 | np.ndarray |

## AlignedTimeseries

|  |  |  |
| --- | --- | --- |
| 示例 | 操作 | 返回类型 |
| `data.timestamps` | 时间戳数组 | `np.ndarray` |
| `data.values` | 值矩阵 | `np.ndarray, shape=(N, M)` |
| `data.series_names` | 序列名列表 | `List[str]` |
| `data.shape` | 形状 | `(N, M)` N 是时间戳的数量，M 是序列的数量 |
| `len(data)` | 行数 | `int` |
| `data[0]`、`data[0:10]`、`data[0, 1]` | 行 / 元素索引 | `np.ndarray` / scalar |
| `print(data)`、`data.show(50)` | 格式化输出 | 自动截断的表格 |

# Series Path 命名规则

## 表模型

TsFile 中的时间序列由 **表名 + 标签值 + 物理量** 唯一确定，映射为扁平字符串：

```Plain Text
{table_name}.{tag_value_1}.{tag_value_2}...{field_column_name}
```

示例：

- `weather.Beijing.humidity` — 表: weather, 标签: Beijing, 物理量: humidity
- `sensor.s1.pressure` — 表: sensor, 标签: s1, 物理量: pressure

## 树模型

TsFile 树模型中的时间序列由 一个 root 前缀 + 多级路径段（设备路径） + 物理量 唯一确定，映射为扁平字符串：

```Plain Text
root.{seg_1}.{seg_2}...{seg_k}.{field}
```

约束：

- 必须以固定根 `root.` 开头
- 中间 `{seg_1}..{seg_k}` 是设备路径段，段数 `k ≥ 1`，不同设备深度可以不同。
- 末段 `{field}` 是物理量名
- 各段之间用 `.` 分隔。

示例：

- `root.db.d1.humidity` — 根: `root`，设备路径: `db.d1`，物理量: `humidity`
- `root.db.d1.temperature` — 根: `root`，设备路径: `db.d1`，物理量: `temperature`
- `root.sg2.pressure` — 根: `root`，设备路径: `sg2`，物理量: `pressure`

# 基本操作

## 加载

树表不允许混用：一个目录 / 文件列表中**不允许**同时出现表模型和树模型的 TsFile，检测到时加载阶段抛异常

支持文件路径、文件列表或目录：

```Python
from tsfile import TsFileDataFrame

df = TsFileDataFrame(["data/weather.tsfile", "data/sensor.tsfile"])
df = TsFileDataFrame("data/")  # 递归查找目录下所有 .tsfile
print(df)
```

### df 的展示结果

- `print(df)` — 输出多行时间序列元信息（最大行数固定为 20，超出头尾截断）
- `df.show(max_rows=20)`**—**作用与 print 基本一致，但是可以控制最大输出行数

**表模型与树模型输出差异**：

**`print(df)`****：**

|  |  |
| --- | --- |
| 模型 | 表头 |
| 表 | `index │ table │ <tag1> │ <tag2> │ ... │ field │ start_time │ end_time │ count` |
| 树 | `index │ _col_1 │ _col_2 │ ... │ _col_N │ field │ start_time │ end_time │ count` |

- 不同层级深度的设备：路径段**按左对齐**，短路径在 `_col_i` 末尾补 `None`（与表模型短 tag 补 `None` 同规则）

#### 表模型

```JavaScript
TsFileDataFrame(table model, 972 time series, 5 files)
     table  ps_id                    sn  frac                                   field           start_time             end_time  count
  0    pvf     10  30100194A00234H00572     1                                     pac  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  1    pvf     10  30100194A00234H00572     1                      tenmeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  2    pvf     10  30100194A00234H00572     1                  tenmeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  3    pvf     10  30100194A00234H00572     1                   eightymeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  4    pvf     10  30100194A00234H00572     1               eightymeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  5    pvf     10  30100194A00234H00572     1      onehundredandtwentymeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  6    pvf     10  30100194A00234H00572     1  onehundredandtwentymeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  7    pvf     10  30100194A00234H00572     1                         totalcloudcover  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  8    pvf     10  30100194A00234H00572     1                         surfacepressure  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  9    pvf     10  30100194A00234H00572     1                              irradiance  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
...
962    pvf  10044  GCBZT02500A231209186     1                  tenmeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
963    pvf  10044  GCBZT02500A231209186     1                   eightymeterswindspeed  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
964    pvf  10044  GCBZT02500A231209186     1               eightymeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
965    pvf  10044  GCBZT02500A231209186     1      onehundredandtwentymeterswindspeed  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
966    pvf  10044  GCBZT02500A231209186     1  onehundredandtwentymeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
967    pvf  10044  GCBZT02500A231209186     1                         totalcloudcover  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
968    pvf  10044  GCBZT02500A231209186     1                         surfacepressure  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
969    pvf  10044  GCBZT02500A231209186     1                              irradiance  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
970    pvf  10044  GCBZT02500A231209186     1                      scatteredradiation  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
971    pvf  10044  GCBZT02500A231209186     1                         directradiation  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
```

#### 树模型

```JavaScript
TsFileDataFrame(tree model, 972 time series, 5 files)
     _col_1                _col_2 _col_3                                     field           start_time             end_time  count
  0      10  30100194A00234H00572      1                                       pac  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  1      10  30100194A00234H00572      1                        tenmeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  2      10  30100194A00234H00572      1                    tenmeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  3      10  30100194A00234H00572      1                     eightymeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  4      10  30100194A00234H00572      1                 eightymeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  5      10  30100194A00234H00572      1        onehundredandtwentymeterswindspeed  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  6      10  30100194A00234H00572      1    onehundredandtwentymeterswinddirection  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  7      10  30100194A00234H00572      1                           totalcloudcover  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  8      10  30100194A00234H00572      1                           surfacepressure  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
  9      10  30100194A00234H00572      1                                irradiance  2024-04-02 00:00:00  2024-10-28 23:45:00  20160
...
962   10044  GCBZT02500A231209186      1                    tenmeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
963   10044  GCBZT02500A231209186      1                     eightymeterswindspeed  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
964   10044  GCBZT02500A231209186      1                 eightymeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
965   10044  GCBZT02500A231209186      1        onehundredandtwentymeterswindspeed  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
966   10044  GCBZT02500A231209186      1    onehundredandtwentymeterswinddirection  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
967   10044  GCBZT02500A231209186      1                           totalcloudcover  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
968   10044  GCBZT02500A231209186      1                           surfacepressure  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
969   10044  GCBZT02500A231209186      1                                irradiance  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
970   10044  GCBZT02500A231209186      1                        scatteredradiation  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
971   10044  GCBZT02500A231209186      1                           directradiation  2024-01-23 00:00:00  2024-09-13 23:45:00  22560
```

初始化时只扫描元数据，不读取实际数值。多文件时自动并行加载。

如果多个文件包含同名序列（如不同日期分片的 `weather.Beijing.humidity`），会自动合并为一条连续的时间序列。（对于重复的时间戳，仅保留第一条时间序列，由于这并不是应该出现的情况，因此这种场景下可能会导致元数据的失真，对一个时间戳进行重复统计，尽量在预处理阶段就避免这种情况的发生）

支持 `with` 语句自动释放文件句柄：

```Python
with TsFileDataFrame("data/") as df:
    pass  # 退出后自动关闭
    
tsdf = TsFileDataFrame("data/")
tsdf.close() # 也可以自己关闭
```

## 浏览

### list_timeseries

**表模型与树模型输出差异**：

**`list_timeseries`****：**

|  |  |
| --- | --- |
| 模型 | 序列名 |
| 表 | `weather.beijing.temperature`   (`<table>.<tag_1>...<tag_k>.<field>`) |
| 树 | `root.weather.beijing.temperature`   (`<root>.<seg_1>...<seg_k>.<field>`) |

#### 表模型

```Python
df.list_timeseries("weather")            # 按表名前缀筛选
df.list_timeseries("weather.Beijing")    # 按标签前缀筛选
```

这会列出当前的 TsFileDataFrame 读取的 TsFile 中有哪些序列

```Python
>>> df.list_timeseries("weather")
['weather.Beijing.humidity', 'weather.Beijing.temperature', 'weather.Shanghai.humidity', 'weather.Shanghai.temperature', 'weather.Guangzhou.humidity', 'weather.Guangzhou.temperature']
>>> df.list_timeseries("weather.Beijing")
['weather.Beijing.humidity', 'weather.Beijing.temperature']
```

注："weather" 可以匹配 "weather.beijing.humidity"

#### 树模型

```Python
df.list_timeseries("root.db.d1")            # 按序列名前缀筛选
```

这会列出当前的 TsFileDataFrame 读取的 TsFile 中有哪些序列

```Python
>>> df.list_timeseries()                # 等价于 path_prefix=""，返回全部
['root.db.d1.humidity', 'root.db.d1.temperature',
 'root.db.d2.humidity', 'root.db.d2.temperature',
 'root.db.d3.humidity', 'root.db.d3.temperature']

>>> df.list_timeseries("root.db")
['root.db.d1.humidity', 'root.db.d1.temperature',
 'root.db.d2.humidity', 'root.db.d2.temperature',
 'root.db.d3.humidity', 'root.db.d3.temperature']

>>> df.list_timeseries("root.db.d1")
['root.db.d1.humidity', 'root.db.d1.temperature']
```

注："root.db" 可以匹配 "root.db.d1.humidity"

### list_timeseries_metadata

返回值类型：pandas.DataFrame

#### 表模型

```Python
>>> df.list_timeseries_metadata()
       table       city        field           start_time             end_time  count
0    weather    Beijing     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1    weather    Beijing  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
2    weather   Shanghai     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
3    weather   Shanghai  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
4    weather  Guangzhou     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
5    weather  Guangzhou  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

>>> df.list_timeseries_metadata("weather.Beijing")
       table       city        field           start_time             end_time  count
0    weather    Beijing     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1    weather    Beijing  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

>>> df.list_timeseries_metadata("weather.Beijing").to_csv("beijing_meta.csv")
```

#### 树模型

```Python
>>> df.list_timeseries_metadata()
   _col_1  _col_2        field           start_time             end_time  count
0      db      d1     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1      db      d1  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
2      db      d2     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
3      db      d2  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
4      db      d3     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
5      db      d3  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

>>> df.list_timeseries_metadata("root.db.d1")
   _col_1  _col_2        field           start_time             end_time  count
0      db      d1     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1      db      d1  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
```

- 不传参等价于 `path_prefix=""`，返回全部序列元数据

## 获取时间序列

`df[...]` 返回的是 `Timeseries` 懒加载句柄，不触发数据读取：

### 表模型

```Python
ts = df["weather.Beijing.humidity"]      # 按名称
ts = df[0]                               # 按索引（支持负索引）
sub_df = df[df['city'] == 'Beijing']         # 按元数据过滤

sub_df = df[0:3]                        # 切片 -> TsFileDataFrame(view)
sub_df = df[[0, 2, 5]]                  # 整数列表 -> TsFileDataFrame(view)
```

得到如下结果

```SQL
>>> df["weather.Beijing.humidity"]
Timeseries('weather.Beijing.humidity', count=2880, start=2026-01-27 00:00:00, end=2026-02-05 23:55:00)
>>> df[0]
Timeseries('iot.dev_0000.temp', count=10000, start=2026-01-27 00:00:00, end=2026-01-27 02:46:39)

>>> df[0:3]
TsFileDataFrame(table model, 3 time series, subset of 6)
   table         city        field           start_time             end_time  count
0    iot     dev_0000         temp  2026-01-27 00:00:00  2026-01-27 02:46:39  10000
1    iot     dev_0000     humidity  2026-01-27 00:00:00  2026-01-27 02:46:39  10000
2    iot     dev_0000     pressure  2026-01-27 00:00:00  2026-01-27 02:46:39  10000

>>> df[[0, 2, 5]]
TsFileDataFrame(table model, 3 time series, subset of 6)
   table         city        field           start_time             end_time  count
0    iot     dev_0000         temp  2026-01-27 00:00:00  2026-01-27 02:46:39  10000
2    iot     dev_0000     pressure  2026-01-27 00:00:00  2026-01-27 02:46:39  10000
5    iot     dev_0001         temp  2026-01-27 00:00:00  2026-01-27 02:46:39  10000
```

查看序列元信息（从缓存获取，无 I/O）：

```Python
ts.name     # -> "weather.Beijing.humidity"
len(ts)     # -> 数据点总数
ts.stats    # -> {'start_time': int, 'end_time': int, 'count': int}
```

以weather数据为例，输出如下：

```SQL
>>> ts = df["weather.Beijing.humidity"]
>>> ts.name
'weather.Beijing.humidity'
>>> ts.stats
{'start_time': 1769443200000, 'end_time': 1770306900000, 'count': 2880}
```

### 树模型

```Python
ts = df["root.db.d1.humidity"]           # 按名称（完整树路径）
ts = df[0]                               # 按索引（支持负索引）
sub_df = df[df['_col_2'] == 'd1']            # 按元数据过滤（树模型用 _col_i 替代 tag 列名）

sub_df = df[0:3]                        # 切片 -> TsFileDataFrame（view）
sub_df = df[[0, 2, 5]]                  # 列表 -> TsFileDataFrame（view）
```

得到如下结果

```SQL
>>> df["root.db.d1.humidity"]
Timeseries('root.db.d1.humidity', count=2880, start=2026-01-27 00:00:00, end=2026-02-05 23:55:00)

>>> df[0]
Timeseries('root.db.d1.humidity', count=2880, start=2026-01-27 00:00:00, end=2026-02-05 23:55:00)

>>> df[0:3]
TsFileDataFrame(tree model, 3 time series, subset of 6)
   _col_1  _col_2        field           start_time             end_time  count
0      db      d1     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1      db      d1  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
2      db      d2     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

>>> df[[0, 2, 5]]
TsFileDataFrame(tree model, 3 time series, subset of 6)
   _col_1  _col_2        field           start_time             end_time  count
0      db      d1     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
2      db      d2     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
5      db      d3  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

>>> df[df['_col_2'] == 'd1']
TsFileDataFrame(tree model, 2 time series, subset of 6)
   _col_1  _col_2        field           start_time             end_time  count
0      db      d1     humidity  2026-01-27 00:00:00  2026-02-05 23:55:00   2880
1      db      d1  temperature  2026-01-27 00:00:00  2026-02-05 23:55:00   2880

```

查看序列元信息（从缓存获取，无 I/O）：

```Python
>>> ts = df["root.db.d1.humidity"]
>>> ts.name
'root.db.d1.humidity'
>>> len(ts)
2880
>>> ts.stats
{'start_time': 1769443200000, 'end_time': 1770306900000, 'count': 2880}
```

## 读取数据

对 `Timeseries` 按行号索引时才触发实际的文件读取：

```Python
val = ts[20]         # -> float
window = ts[20:100]  # -> np.ndarray, shape=(80,)
last_ten = ts[-10:]  # -> np.ndarray
sampled = ts[::2]    # -> np.ndarray（步长采样）
ts.timestamps[20:100]  # -> 对应行号的时间戳, np.ndarray
```

结果展示如下，访问会触发一次磁盘访问（会有缓存机制来避免重复读取）

```SQL
>>> ts[20]
46.1
>>> ts[20:100]
array([46.1 , 41.72, 52.94, 55.59, 67.34, 88.48, 86.97, 69.89, 44.42,
       42.26, 59.43, 81.44, 54.05, 47.05, 43.73, 78.61, 40.28, 75.34,
       78.56, 57.92, 83.16, 56.54, 55.55, 76.48, 84.36, 45.98, 78.04,
       78.55, 66.14, 41.27, 41.57, 55.72, 85.38, 60.52, 51.44, 54.49,
       86.48, 71.67, 80.18, 84.63, 80.37, 55.9 , 51.4 , 80.9 , 40.35,
       60.87, 45.99, 87.15, 65.94, 58.18, 88.12, 64.86, 54.24, 70.48,
       42.57, 85.41, 47.24, 89.28, 73.61, 51.88, 58.39, 71.68, 44.51,
       56.04, 42.04, 73.88, 65.6 , 72.26, 74.55, 86.84, 57.05, 86.23,
       52.9 , 80.86, 66.48, 44.66, 85.02, 56.95, 76.3 , 84.35])
>>> ts[-10:]
array([45.62, 43.21, 53.85, 54.13, 82.25, 73.78, 83.  , 54.47, 62.72,
       81.35])
>>> ts[::2]
array([58.73, 47.8 , 70.06, ..., 82.25, 83.  , 62.72], shape=(1440,))
>>> ts.timestamps[20:100]
array([1769449200000, 1769449500000, 1769449800000, 1769450100000,
       1769450400000, 1769450700000, 1769451000000, 1769451300000,
       1769451600000, 1769451900000, 1769452200000, 1769452500000,
       1769452800000, 1769453100000, 1769453400000, 1769453700000,
       1769454000000, 1769454300000, 1769454600000, 1769454900000,
       1769455200000, 1769455500000, 1769455800000, 1769456100000,
       1769456400000, 1769456700000, 1769457000000, 1769457300000,
       1769457600000, 1769457900000, 1769458200000, 1769458500000,
       1769458800000, 1769459100000, 1769459400000, 1769459700000,
       1769460000000, 1769460300000, 1769460600000, 1769460900000,
       1769461200000, 1769461500000, 1769461800000, 1769462100000,
       1769462400000, 1769462700000, 1769463000000, 1769463300000,
       1769463600000, 1769463900000, 1769464200000, 1769464500000,
       1769464800000, 1769465100000, 1769465400000, 1769465700000,
       1769466000000, 1769466300000, 1769466600000, 1769466900000,
       1769467200000, 1769467500000, 1769467800000, 1769468100000,
       1769468400000, 1769468700000, 1769469000000, 1769469300000,
       1769469600000, 1769469900000, 1769470200000, 1769470500000,
       1769470800000, 1769471100000, 1769471400000, 1769471700000,
       1769472000000, 1769472300000, 1769472600000, 1769472900000])
```

## 多序列对齐查询

当需要多条序列在同一时间轴上严格对齐时，使用 `.``loc`：

### 表模型

```Python
data = df.loc[start_time:end_time, [
    "weather.Beijing.humidity",
    "weather.Beijing.temperature",
    "sensor.s1.pressure",
]]
```

返回的 `AlignedTimeseries` 将所有序列对齐到时间戳并集，缺失位置填充 NaN：

```Python
data.timestamps    # np.ndarray，毫秒时间戳
data.values        # np.ndarray, shape=(N, 3)
data.series_names  # ["weather.Beijing.humidity", ...]
data.shape         # (N, 3)
data[0:10]         # 前 10 行, np.ndarray shape=(10, 3)
print(data)        # 截断显示
data.show(50)      # 显示最多 50 行
```

序列指定支持名称和索引混用：

```Python
df.loc[start_time:end_time, [0, 1, 4]]
```

演示输出如下，实现中为对齐提供了一种更加可读的展示方式

```SQL
>>> df.loc[1769616000000:1769702100000,['weather.Beijing.temperature','weather.Beijing.humidity','sensor.s2.pressure']]
AlignedTimeseries(288 rows, 3 series)
          timestamp  weather.Beijing.temperature  weather.Beijing.humidity  sensor.s2.pressure
2026-01-29 00:00:00                        29.12                     92.87                 NaN
2026-01-29 00:05:00                         1.55                     87.34                 NaN
2026-01-29 00:10:00                         7.78                     38.18                 NaN
2026-01-29 00:15:00                         7.72                     58.50                 NaN
2026-01-29 00:20:00                         4.23                     64.46                 NaN
2026-01-29 00:25:00                        10.22                     59.47                 NaN
2026-01-29 00:30:00                        -1.46                     47.32                 NaN
2026-01-29 00:35:00                        20.83                     79.33                 NaN
2026-01-29 00:40:00                         6.36                     92.01                 NaN
2026-01-29 00:45:00                         5.40                     89.47                 NaN
...
2026-01-29 23:10:00                        20.98                     52.73                 NaN
2026-01-29 23:15:00                         0.63                     65.57                 NaN
2026-01-29 23:20:00                        -5.77                     40.72                 NaN
2026-01-29 23:25:00                        15.92                     73.06                 NaN
2026-01-29 23:30:00                        -1.71                     87.91                 NaN
2026-01-29 23:35:00                        21.07                     63.98                 NaN
2026-01-29 23:40:00                        -9.30                     52.59                 NaN
2026-01-29 23:45:00                        -4.54                     53.19                 NaN
2026-01-29 23:50:00                         3.39                     60.36                 NaN
2026-01-29 23:55:00                         1.66                     32.94                 NaN

>>> df.loc[1769616000000:1769702100000,['weather.Beijing.temperature','weather.Beijing.humidity','sensor.s2.pressure']][10:11]
array([[19.43, 48.3 ,   nan]])
```

### 树模型

```Python
# 树模型示例
data = df.loc[start_time:end_time, [
    "root.db.d1.humidity",
    "root.db.d1.temperature",
    "root.db.d2.pressure",
]]
```

返回的 `AlignedTimeseries` 将所有序列对齐到时间戳并集，缺失位置填充 NaN：

```Python
data.timestamps    # np.ndarray，毫秒时间戳
data.values        # np.ndarray, shape=(N, 3)
data.series_names  # ["root.db.d1.humidity", "root.db.d1.temperature", "root.db.d2.pressure"]
data.shape         # (N, 3)
data[0:10]         # 前 10 行, np.ndarray shape=(10, 3)
print(data)        # 截断显示
data.show(50)      # 显示最多 50 行
```

序列指定支持名称和索引混用：

```Python
df.loc[start_time:end_time, [0, 1, 4]]
df.loc[start_time:end_time, [0, "root.db.d1.temperature", 4]]
```

演示输出如下，实现中为对齐提供了一种更加可读的展示方式

```Python
>>> df.loc[1769616000000:1769702100000,['root.db.d1.temperature','root.db.d1.humidity','root.db.d2.pressure']]
AlignedTimeseries(288 rows, 3 series)
          timestamp  root.db.d1.temperature  root.db.d1.humidity  root.db.d2.pressure
2026-01-29 00:00:00                   29.12                92.87                  NaN
2026-01-29 00:05:00                    1.55                87.34                  NaN
2026-01-29 00:10:00                    7.78                38.18                  NaN
2026-01-29 00:15:00                    7.72                58.50                  NaN
2026-01-29 00:20:00                    4.23                64.46                  NaN
2026-01-29 00:25:00                   10.22                59.47                  NaN
2026-01-29 00:30:00                   -1.46                47.32                  NaN
2026-01-29 00:35:00                   20.83                79.33                  NaN
2026-01-29 00:40:00                    6.36                92.01                  NaN
2026-01-29 00:45:00                    5.40                89.47                  NaN
...
2026-01-29 23:10:00                   20.98                52.73                  NaN
2026-01-29 23:15:00                    0.63                65.57                  NaN
2026-01-29 23:20:00                   -5.77                40.72                  NaN
2026-01-29 23:25:00                   15.92                73.06                  NaN
2026-01-29 23:30:00                   -1.71                87.91                  NaN
2026-01-29 23:35:00                   21.07                63.98                  NaN
2026-01-29 23:40:00                   -9.30                52.59                  NaN
2026-01-29 23:45:00                   -4.54                53.19                  NaN
2026-01-29 23:50:00                    3.39                60.36                  NaN
2026-01-29 23:55:00                    1.66                32.94                  NaN

# 这种输出方式只能输出值列，如果想要输出时间戳列需要  df.loc[...].timestamps[10:11]
>>> df.loc[1769616000000:1769702100000,['root.db.d1.temperature','root.db.d1.humidity','root.db.d2.pressure']][10:11]
array([[19.43, 48.3 ,   nan]])
```

# 当前已知限制

**时间戳缓存**：底层 TsFile 仅支持 time range 查询，因此初始化时会为每条序列缓存完整的时间戳数组，用于行号到时间戳的映射（也会用于计算统计信息，因为底层Tsfile还没统计信息的接口）。这意味着：

- 初始化时需要一次全量时间戳扫描，数据量大时会有加载延迟
- 时间戳数组常驻内存（实际数值数据不缓存）

如果 TsFile 后续支持基于行号的查询接口，以及序列元数据的获取接口，可消除此开销。

# 性能特性

- **并行加载**：多文件时自动使用线程池并行扫描元数据（Worker 数 = min(文件数, CPU 核数)）。需注意文件数不超过系统最大文件句柄数限制。
- **对齐查询瓶颈**：`.loc` 的时间对齐在 Python 层完成（合并去重 + NaN 填充），对于数千万时间戳的极大规模查询可能成为瓶颈。
- TO BE ADDED

# 场景

## 场景一：时序大模型预训练

### **数据格式**

HuggingFace Arrow 格式（`datasets.load_from_disk`），每条时序数据是一个字典, 字典的target列是一条完整时序存为 `list<float>` 列。

### 读取流程

```Python
# 初始化阶段from datasets import load_from_disk
import pyarrow.compute as pc

dataset = load_from_disk("/path/to/huggingface_dataset")

# 获取不同序列的长度
lengths = pc.list_value_length(dataset.data.column('target'))
probs = lengths.to_pylist() / np.sum(lengths.to_pylist())

# __getitem__
def __getitem__(self, index):
    idx = np.random.choice(len(self.probs), p=self.probs)
    series = np.array(self.data[idx]['target'])  # ← 读取整条序列 L 个点
    start = np.random.randint(0, len(series) - self.context_length)
    return series[start : start + self.context_length]  # 只用 W 个点
```

*Arrow**`list<float>`**列在磁盘上是扁平连续 buffer，访问**`dataset[i]['target']`**必须读出整个 list 的所有字节，****没有 API 可以只读 list 中间的一段子区间****。*

场景接口需求

1. 获取不同的序列长度
2. 

## 等价写法

```Python
tsdf = TsFileDataFrame("/path/to/tsfiles/")

mdata = tsdf.metadata()
# __getitem__：只读窗口所在的 Chunk/Pagedef __getitem__(self, index):
series_name, start_offset = self.window_index[index]
ts = tsdf[series_name]
return ts[start_offset : start_offset + self.context_length]
```

## 场景二：南航飞行数据（Parquet）

### **数据格式**

宽表 Parquet，每行是一个时间步的所有传感器读数（行式时间步，列式传感器），以 row group 为物理分组单元。

### 读取流程

```Python
# 初始化阶段：手写 row group 寻址（~10 行）from pyarrow.parquet import ParquetFile, read_metadata

parquet_file = ParquetFile(parquet_path)
metadata = read_metadata(parquet_path)
total_rows = metadata.num_rows
num_row_groups = metadata.num_row_groups

# 手动构建偏移表（用户自己实现"时间索引"）
row_group_offsets = []
offset = 0
for i in range(num_row_groups):
    row_group_offsets.append(offset)
    offset += metadata.row_group(i).num_rows

# __getitem__：手写窗口跨 row group 处理（~20 行）def __getitem__(self, idx):
def __getitem__(self, idx):
    start_row = idx * self.sel_len
    end_row = (idx + 1) * self.sel_len

    # 找覆盖 [start_row, end_row] 的所有 row groups
    row_groups_to_read = []
    for i, rg_offset in enumerate(self.row_group_offsets):
        rg_end = rg_offset + metadata.row_group(i).num_rows
        if rg_offset < end_row and rg_end > start_row:
            row_groups_to_read.append(i)

    # 读多个 row group，截取精确行范围
    table = self.parquet_file.read_row_groups(row_groups_to_read)
    local_start = start_row - self.row_group_offsets[row_groups_to_read[0]]
    return table.slice(local_start, self.sel_len).to_pandas().to_numpy()
```

*手写维护成本高，且Parquet row group 是物理分区单元，与时间序列的逻辑分组（设备、传感器）没有对应关系。一个训练窗口可能横跨两个 row group，必须读取超出需要的数据再截取。*

## 场景三：协鑫光伏预测

### 数据格式：

任意可以被全量加载到内存DataFrame 的数据

```Python
# 全量加载进 DataFrame
df_raw = pd.read_csv(self.dataset_file_path)
data = df_raw[self.target_columns].values  # 全量 numpy 数组# 按 70/10/20 比例切分
train_data = data[border1s[0]:border2s[0]]
self.mean = np.mean(train_data, axis=0)
self.std = np.std(train_data, axis=0) + 1e-5
data = (data - self.mean) / self.std

self.data_x = data[border1:border2].astype(float)
# __getitem__ 是纯 numpy 切片，快，但要求数据全在内存
```

和csv一样，做全量加载

```Python
# 调用 TsFile API，但立刻转为全量 DataFrame
df_raw = to_dataframe(file_path)  # ← TsFile 读取
data = df_raw.values              # ← 转全量 numpy，与 CSV 版完全相同的后续处理
```

*基本都是 DataFrame 原生的操作，除了值过滤外，在 TsFileDataFrame 都有对应操作*
