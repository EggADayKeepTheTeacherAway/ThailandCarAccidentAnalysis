## Import libraries


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import requests
import holidays
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,LSTM, Dense, Dropout
```

## Load and Prepare Data


```python
# Path the local folder
data_folder = '../../data'

# Find all CSVs starting with 'vehicle_accident' in that folder
file_list = glob.glob(os.path.join(data_folder, 'vehicle_accident *.csv'))

# Load all files into a list of DataFrames
dfs = []
for file in file_list:
    df = pd.read_csv(file, encoding='utf-8', low_memory=False)
    dfs.append(df)

# Concatenate all DataFrames
df = pd.concat(dfs, ignore_index=True)
df.head(10)
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
      <th>ปีที่เกิดเหตุ</th>
      <th>วันที่เกิดเหตุ</th>
      <th>เวลา</th>
      <th>วันที่รายงาน</th>
      <th>เวลาที่รายงาน</th>
      <th>ACC_CODE</th>
      <th>หน่วยงาน</th>
      <th>สายทางหน่วยงาน</th>
      <th>รหัสสายทาง</th>
      <th>สายทาง</th>
      <th>...</th>
      <th>รถบรรทุก 6 ล้อ</th>
      <th>รถบรรทุกมากกว่า 6 ล้อ ไม่เกิน 10 ล้อ</th>
      <th>รถบรรทุกมากกว่า 10 ล้อ (รถพ่วง)</th>
      <th>รถอีแต๋น</th>
      <th>อื่นๆ</th>
      <th>คนเดินเท้า</th>
      <th>จำนวนผู้เสียชีวิต</th>
      <th>จำนวนผู้บาดเจ็บสาหัส</th>
      <th>จำนวนผู้บาดเจ็บเล็กน้อย</th>
      <th>รวมจำนวนผู้บาดเจ็บ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:45</td>
      <td>6566872</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>ชน.5016</td>
      <td>เทศบาลตำบลวัดสิงห์ - บ้านน้ำพุ (ช่วงหันคา)</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:44</td>
      <td>6566880</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>มค.4012</td>
      <td>แยกทางหลวงหมายเลข 2152 (กม.ที่ 31+700) - บ้านก...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:03</td>
      <td>09/02/2022</td>
      <td>08:41</td>
      <td>5706553</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4</td>
      <td>พ่อตาหินช้าง - วังครก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>02/01/2022</td>
      <td>06:21</td>
      <td>5485750</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4030</td>
      <td>ถลาง - หาดราไวย์</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>24/01/2022</td>
      <td>09:59</td>
      <td>5624452</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>216</td>
      <td>ถนนวงแหวนรอบเมืองอุดรธานีด้านทิศตะวันออก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>02/01/2022</td>
      <td>11:46</td>
      <td>6566842</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>อย.4009</td>
      <td>แยกทางหลวงหมายเลข 3111 (กม.ที่ 19+200) - บ้านป...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:08</td>
      <td>03/03/2022</td>
      <td>10:14</td>
      <td>5836781</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>3477</td>
      <td>บางปะอิน - เกาะเรียน</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:10</td>
      <td>22/02/2022</td>
      <td>14:01</td>
      <td>5783258</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>3438</td>
      <td>ดินแดง - ไผ่งาม</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:10</td>
      <td>02/01/2022</td>
      <td>06:45</td>
      <td>6566818</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>ชม.3059</td>
      <td>แยกทางหลวงหมายเลข 107 (กม.ที่ 152+300) - เขื่อ...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:18</td>
      <td>02/01/2022</td>
      <td>06:44</td>
      <td>6566847</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>พง.5012</td>
      <td>เชื่อมถนนเทศบาลท้ายเหมือง - ชายทะเลท้ายเหมือง</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 61 columns</p>
</div>




```python
# Convert "วันที่เกิดเหตุ" and "เวลา" to datetime
df['datetime'] = pd.to_datetime(df['วันที่เกิดเหตุ'] + ' ' + df['เวลา'], format='%d/%m/%Y %H:%M')
df.dropna(subset=['LATITUDE', 'LONGITUDE', 'datetime','สภาพอากาศ'], inplace=True)
df
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
      <th>ปีที่เกิดเหตุ</th>
      <th>วันที่เกิดเหตุ</th>
      <th>เวลา</th>
      <th>วันที่รายงาน</th>
      <th>เวลาที่รายงาน</th>
      <th>ACC_CODE</th>
      <th>หน่วยงาน</th>
      <th>สายทางหน่วยงาน</th>
      <th>รหัสสายทาง</th>
      <th>สายทาง</th>
      <th>...</th>
      <th>รถบรรทุกมากกว่า 6 ล้อ ไม่เกิน 10 ล้อ</th>
      <th>รถบรรทุกมากกว่า 10 ล้อ (รถพ่วง)</th>
      <th>รถอีแต๋น</th>
      <th>อื่นๆ</th>
      <th>คนเดินเท้า</th>
      <th>จำนวนผู้เสียชีวิต</th>
      <th>จำนวนผู้บาดเจ็บสาหัส</th>
      <th>จำนวนผู้บาดเจ็บเล็กน้อย</th>
      <th>รวมจำนวนผู้บาดเจ็บ</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:45</td>
      <td>6566872</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>ชน.5016</td>
      <td>เทศบาลตำบลวัดสิงห์ - บ้านน้ำพุ (ช่วงหันคา)</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:44</td>
      <td>6566880</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>มค.4012</td>
      <td>แยกทางหลวงหมายเลข 2152 (กม.ที่ 31+700) - บ้านก...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:03</td>
      <td>09/02/2022</td>
      <td>08:41</td>
      <td>5706553</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4</td>
      <td>พ่อตาหินช้าง - วังครก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-01-01 00:03:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>02/01/2022</td>
      <td>06:21</td>
      <td>5485750</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4030</td>
      <td>ถลาง - หาดราไวย์</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2022-01-01 00:05:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>24/01/2022</td>
      <td>09:59</td>
      <td>5624452</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>216</td>
      <td>ถนนวงแหวนรอบเมืองอุดรธานีด้านทิศตะวันออก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2022-01-01 00:05:00</td>
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
      <th>226105</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>13/01/2014</td>
      <td>13:33</td>
      <td>3576908</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226106</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:08</td>
      <td>3578962</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>41</td>
      <td>สวนสมบูรณ์ - เกาะมุกข์</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226107</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:29</td>
      <td>3579135</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>202</td>
      <td>แก้งสนามนาง - ดอนตะหนิน</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226108</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:25</td>
      <td>06/01/2014</td>
      <td>11:44</td>
      <td>3576181</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4170</td>
      <td>สระเกศ - หัวถนน</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2013-12-31 23:25:00</td>
    </tr>
    <tr>
      <th>226109</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:30</td>
      <td>02/01/2014</td>
      <td>17:33</td>
      <td>3476271</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>นศ.3053</td>
      <td>แยกทางหลวงหมายเลข 408 (กม.ที่ 48+900) - เขตเทศ...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2013-12-31 23:30:00</td>
    </tr>
  </tbody>
</table>
<p>224613 rows × 62 columns</p>
</div>




```python
# Round down the minute to make it a flat hour store as new column named "datetime_hour"
df['datetime_hour'] = df['datetime'].dt.floor('h')
df
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
      <th>ปีที่เกิดเหตุ</th>
      <th>วันที่เกิดเหตุ</th>
      <th>เวลา</th>
      <th>วันที่รายงาน</th>
      <th>เวลาที่รายงาน</th>
      <th>ACC_CODE</th>
      <th>หน่วยงาน</th>
      <th>สายทางหน่วยงาน</th>
      <th>รหัสสายทาง</th>
      <th>สายทาง</th>
      <th>...</th>
      <th>รถบรรทุกมากกว่า 10 ล้อ (รถพ่วง)</th>
      <th>รถอีแต๋น</th>
      <th>อื่นๆ</th>
      <th>คนเดินเท้า</th>
      <th>จำนวนผู้เสียชีวิต</th>
      <th>จำนวนผู้บาดเจ็บสาหัส</th>
      <th>จำนวนผู้บาดเจ็บเล็กน้อย</th>
      <th>รวมจำนวนผู้บาดเจ็บ</th>
      <th>datetime</th>
      <th>datetime_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:45</td>
      <td>6566872</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>ชน.5016</td>
      <td>เทศบาลตำบลวัดสิงห์ - บ้านน้ำพุ (ช่วงหันคา)</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
      <td>2022-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:44</td>
      <td>6566880</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>มค.4012</td>
      <td>แยกทางหลวงหมายเลข 2152 (กม.ที่ 31+700) - บ้านก...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
      <td>2022-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:03</td>
      <td>09/02/2022</td>
      <td>08:41</td>
      <td>5706553</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4</td>
      <td>พ่อตาหินช้าง - วังครก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2022-01-01 00:03:00</td>
      <td>2022-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>02/01/2022</td>
      <td>06:21</td>
      <td>5485750</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4030</td>
      <td>ถลาง - หาดราไวย์</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2022-01-01 00:05:00</td>
      <td>2022-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>24/01/2022</td>
      <td>09:59</td>
      <td>5624452</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>216</td>
      <td>ถนนวงแหวนรอบเมืองอุดรธานีด้านทิศตะวันออก</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2022-01-01 00:05:00</td>
      <td>2022-01-01 00:00:00</td>
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
      <th>226105</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>13/01/2014</td>
      <td>13:33</td>
      <td>3576908</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226106</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:08</td>
      <td>3578962</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>41</td>
      <td>สวนสมบูรณ์ - เกาะมุกข์</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226107</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:29</td>
      <td>3579135</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>202</td>
      <td>แก้งสนามนาง - ดอนตะหนิน</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226108</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:25</td>
      <td>06/01/2014</td>
      <td>11:44</td>
      <td>3576181</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4170</td>
      <td>สระเกศ - หัวถนน</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2013-12-31 23:25:00</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
    <tr>
      <th>226109</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:30</td>
      <td>02/01/2014</td>
      <td>17:33</td>
      <td>3476271</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>นศ.3053</td>
      <td>แยกทางหลวงหมายเลข 408 (กม.ที่ 48+900) - เขตเทศ...</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2013-12-31 23:30:00</td>
      <td>2013-12-31 23:00:00</td>
    </tr>
  </tbody>
</table>
<p>224613 rows × 63 columns</p>
</div>




```python
# One-hot encode "สภาพอากาศ"
weather_dummies = pd.get_dummies(df['สภาพอากาศ'], prefix='weather')
df = pd.concat([df, weather_dummies], axis=1)
df = df.drop('สภาพอากาศ', axis=1)
df
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
      <th>ปีที่เกิดเหตุ</th>
      <th>วันที่เกิดเหตุ</th>
      <th>เวลา</th>
      <th>วันที่รายงาน</th>
      <th>เวลาที่รายงาน</th>
      <th>ACC_CODE</th>
      <th>หน่วยงาน</th>
      <th>สายทางหน่วยงาน</th>
      <th>รหัสสายทาง</th>
      <th>สายทาง</th>
      <th>...</th>
      <th>รวมจำนวนผู้บาดเจ็บ</th>
      <th>datetime</th>
      <th>datetime_hour</th>
      <th>weather_ดินถล่ม</th>
      <th>weather_ฝนตก</th>
      <th>weather_ภัยธรรมชาติ เช่น พายุ น้ำท่วม</th>
      <th>weather_มีหมอก/ควัน/ฝุ่น</th>
      <th>weather_มืดครึ้ม</th>
      <th>weather_อื่นๆ</th>
      <th>weather_แจ่มใส</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:45</td>
      <td>6566872</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>ชน.5016</td>
      <td>เทศบาลตำบลวัดสิงห์ - บ้านน้ำพุ (ช่วงหันคา)</td>
      <td>...</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:01</td>
      <td>02/01/2022</td>
      <td>11:44</td>
      <td>6566880</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>มค.4012</td>
      <td>แยกทางหลวงหมายเลข 2152 (กม.ที่ 31+700) - บ้านก...</td>
      <td>...</td>
      <td>1</td>
      <td>2022-01-01 00:01:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:03</td>
      <td>09/02/2022</td>
      <td>08:41</td>
      <td>5706553</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4</td>
      <td>พ่อตาหินช้าง - วังครก</td>
      <td>...</td>
      <td>0</td>
      <td>2022-01-01 00:03:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>02/01/2022</td>
      <td>06:21</td>
      <td>5485750</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4030</td>
      <td>ถลาง - หาดราไวย์</td>
      <td>...</td>
      <td>1</td>
      <td>2022-01-01 00:05:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022</td>
      <td>01/01/2022</td>
      <td>00:05</td>
      <td>24/01/2022</td>
      <td>09:59</td>
      <td>5624452</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>216</td>
      <td>ถนนวงแหวนรอบเมืองอุดรธานีด้านทิศตะวันออก</td>
      <td>...</td>
      <td>2</td>
      <td>2022-01-01 00:05:00</td>
      <td>2022-01-01 00:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
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
      <th>226105</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>13/01/2014</td>
      <td>13:33</td>
      <td>3576908</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>9</td>
      <td>NaN</td>
      <td>...</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>226106</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:08</td>
      <td>3578962</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>41</td>
      <td>สวนสมบูรณ์ - เกาะมุกข์</td>
      <td>...</td>
      <td>1</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>226107</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:00</td>
      <td>04/02/2014</td>
      <td>12:29</td>
      <td>3579135</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>202</td>
      <td>แก้งสนามนาง - ดอนตะหนิน</td>
      <td>...</td>
      <td>0</td>
      <td>2013-12-31 23:00:00</td>
      <td>2013-12-31 23:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>226108</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:25</td>
      <td>06/01/2014</td>
      <td>11:44</td>
      <td>3576181</td>
      <td>กรมทางหลวง</td>
      <td>ทางหลวง</td>
      <td>4170</td>
      <td>สระเกศ - หัวถนน</td>
      <td>...</td>
      <td>1</td>
      <td>2013-12-31 23:25:00</td>
      <td>2013-12-31 23:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>226109</th>
      <td>2013</td>
      <td>31/12/2013</td>
      <td>23:30</td>
      <td>02/01/2014</td>
      <td>17:33</td>
      <td>3476271</td>
      <td>กรมทางหลวงชนบท</td>
      <td>ทางหลวงชนบท</td>
      <td>นศ.3053</td>
      <td>แยกทางหลวงหมายเลข 408 (กม.ที่ 48+900) - เขตเทศ...</td>
      <td>...</td>
      <td>2</td>
      <td>2013-12-31 23:30:00</td>
      <td>2013-12-31 23:00:00</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>224613 rows × 69 columns</p>
</div>




```python
weather_cols = [col for col in df.columns if col.startswith('weather_')]
# Seperate each lat and lon into a Zone
LAT_GRID_SIZE = 0.1
LON_GRID_SIZE = 0.1
df['lat_zone'] = (df['LATITUDE'] // LAT_GRID_SIZE).astype(int)
df['lon_zone'] = (df['LONGITUDE'] // LON_GRID_SIZE).astype(int)
df['zone_id'] = df['lat_zone'].astype(str) + '_' + df['lon_zone'].astype(str)

# Group by zone, datetime_hour and weather one-hot encoding and mark it as an accident reccord (accident = 1)
df_grouped = df.groupby(['zone_id', 'datetime_hour'] + weather_cols).size().reset_index(name='accident_count')
df_grouped['accident'] = 1
df_grouped.head(5)
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
      <th>zone_id</th>
      <th>datetime_hour</th>
      <th>weather_ดินถล่ม</th>
      <th>weather_ฝนตก</th>
      <th>weather_ภัยธรรมชาติ เช่น พายุ น้ำท่วม</th>
      <th>weather_มีหมอก/ควัน/ฝุ่น</th>
      <th>weather_มืดครึ้ม</th>
      <th>weather_อื่นๆ</th>
      <th>weather_แจ่มใส</th>
      <th>accident_count</th>
      <th>accident</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1_180</td>
      <td>2019-12-29 03:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100_986</td>
      <td>2013-02-10 17:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100_986</td>
      <td>2013-04-13 01:00:00</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100_986</td>
      <td>2013-04-17 10:00:00</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100_986</td>
      <td>2013-04-29 15:00:00</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add back location and datetime features
df_grouped['LATITUDE'] = df_grouped['zone_id'].apply(lambda x: (int(x.split('_')[0]) + 0.5) * LAT_GRID_SIZE)
df_grouped['LONGITUDE'] = df_grouped['zone_id'].apply(lambda x: (int(x.split('_')[1]) + 0.5) * LON_GRID_SIZE)
df_grouped['hour'] = df_grouped['datetime_hour'].dt.hour
df_grouped['dayofweek'] = df_grouped['datetime_hour'].dt.dayofweek
df_grouped['month'] = df_grouped['datetime_hour'].dt.month
```

## Create Negative Samples


```python
number_to_random = len(df_grouped)
rng = np.random.default_rng(seed=42)

negative_df = df_grouped.sample(n=number_to_random, replace=True).copy()
negative_df['datetime_hour'] = negative_df['datetime_hour'] + pd.to_timedelta(rng.integers(1, 1000, size=number_to_random), unit='h')
negative_df['accident'] = 0
negative_df['accident_count'] = 0
```


```python
# Extract datetime features for negative samples
negative_df['LATITUDE'] = negative_df['zone_id'].apply(lambda x: (int(x.split('_')[0]) + 0.5) * LAT_GRID_SIZE)
negative_df['LONGITUDE'] = negative_df['zone_id'].apply(lambda x: (int(x.split('_')[1]) + 0.5) * LON_GRID_SIZE)
negative_df['hour'] = negative_df['datetime_hour'].dt.hour
negative_df['dayofweek'] = negative_df['datetime_hour'].dt.dayofweek
negative_df['month'] = negative_df['datetime_hour'].dt.month
```

## Merge and Preprocessing Data


```python
# Merge postive and negative samples
merged_df = pd.concat([df_grouped, negative_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Select a partial zone base on high number of accident
zone_number = merged_df['zone_id'].value_counts()
danger_zone = zone_number[zone_number > 50].index
merged_df = merged_df[merged_df['zone_id'].isin(danger_zone)].copy()

merged_df.sort_values(['zone_id', 'datetime_hour'], inplace=True)
merged_df['accident_lag1'] = merged_df.groupby('zone_id')['accident'].shift(1).fillna(0)
merged_df['rolling_mean_3'] = merged_df.groupby('zone_id')['accident'].rolling(3).mean().reset_index(0, drop=True).fillna(0)

# Normalize feature
weather_features = [col for col in merged_df.columns if col.startswith('weather_')]
features = ['LATITUDE', 'LONGITUDE', 'hour', 'dayofweek', 'month',
            'accident_lag1', 'rolling_mean_3'] + weather_features

scaler = MinMaxScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])
```

## Create Time Based Sequences for LSTM


```python
def create_sequences(df, time_steps=10):
    df = df.sort_values('datetime_hour')
    X = df[features].values
    y = df['accident'].values
    X_list, y_list = [], []
    for i in range(len(X) - time_steps):
        X_list.append(X[i:i+time_steps])
        y_list.append(y[i+time_steps])
    return X_list, y_list

time_steps = 10
X_seq_all = []
y_seq_all = []
for zone_id in danger_zone:
    zone_df = merged_df[merged_df['zone_id'] == zone_id]
    X_zone, y_zone = create_sequences(zone_df, time_steps=10)
    X_seq_all.extend(X_zone)
    y_seq_all.extend(y_zone)

X_seq = np.array(X_seq_all)
y_seq = np.array(y_seq_all)
```

## Build and Train LSTM Model


```python
# Split train test data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Define a class weight
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
```


```python
# Build the LSTM Model
model = Sequential([
    Input(shape=(X_seq.shape[1], X_seq.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">20,224</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LSTM</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">12,416</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">32,673</span> (127.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">32,673</span> (127.63 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
# Train the LSTM Model
train = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
```

    Epoch 1/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 4ms/step - accuracy: 0.5153 - loss: 0.6926 - val_accuracy: 0.5182 - val_loss: 0.6917
    Epoch 2/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5250 - loss: 0.6915 - val_accuracy: 0.5222 - val_loss: 0.6916
    Epoch 3/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5249 - loss: 0.6913 - val_accuracy: 0.5275 - val_loss: 0.6905
    Epoch 4/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5281 - loss: 0.6904 - val_accuracy: 0.5326 - val_loss: 0.6899
    Epoch 5/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5361 - loss: 0.6892 - val_accuracy: 0.5450 - val_loss: 0.6872
    Epoch 6/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5438 - loss: 0.6869 - val_accuracy: 0.5500 - val_loss: 0.6851
    Epoch 7/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5493 - loss: 0.6853 - val_accuracy: 0.5533 - val_loss: 0.6826
    Epoch 8/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5542 - loss: 0.6830 - val_accuracy: 0.5587 - val_loss: 0.6807
    Epoch 9/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5561 - loss: 0.6819 - val_accuracy: 0.5595 - val_loss: 0.6803
    Epoch 10/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5590 - loss: 0.6811 - val_accuracy: 0.5613 - val_loss: 0.6804
    Epoch 11/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5590 - loss: 0.6802 - val_accuracy: 0.5619 - val_loss: 0.6787
    Epoch 12/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5618 - loss: 0.6795 - val_accuracy: 0.5607 - val_loss: 0.6791
    Epoch 13/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5633 - loss: 0.6788 - val_accuracy: 0.5627 - val_loss: 0.6787
    Epoch 14/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5655 - loss: 0.6778 - val_accuracy: 0.5662 - val_loss: 0.6778
    Epoch 15/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5648 - loss: 0.6778 - val_accuracy: 0.5630 - val_loss: 0.6783
    Epoch 16/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5666 - loss: 0.6769 - val_accuracy: 0.5665 - val_loss: 0.6772
    Epoch 17/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5675 - loss: 0.6758 - val_accuracy: 0.5647 - val_loss: 0.6777
    Epoch 18/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5671 - loss: 0.6765 - val_accuracy: 0.5651 - val_loss: 0.6780
    Epoch 19/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m21s[0m 4ms/step - accuracy: 0.5703 - loss: 0.6758 - val_accuracy: 0.5671 - val_loss: 0.6777
    Epoch 20/20
    [1m4904/4904[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 4ms/step - accuracy: 0.5685 - loss: 0.6757 - val_accuracy: 0.5655 - val_loss: 0.6787


## Evaluate the Model


```python
# Get the predictions
y_pred_prob = model.predict(X_test)

# Convert probabilities to binary (0 or 1) using a threshold
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate F1 Score, Precision and Recall value
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

    [1m2452/2452[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 882us/step
    F1 Score: 0.5426
    Precision: 0.5682
    Recall: 0.5193

