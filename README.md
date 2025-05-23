# Laporan Proyek Machine Learning - Soraya Indah Setiani

## Domain Proyek

Diabetes merupakan salah satu penyakit tidak menular yang sangat berdampak pada kondisi kesehatan masyarakat secara global. Menurut data yang dipublikasi oleh International Diabetes Federation (IDF), penderita diabetes di dunia pada tahun 2021 berada di angka 537 juta jiwa. Angka ini diprediksi akan terus meningkat akibat gaya hidup masyarakat. 
Prediksi untuk mengetahui potensi diabetes pada individu sangat penting untuk dilakukan guna mencegah penyakit diabetes secepat mungkin. Tidak semua masyarakat mampu mengakses fasilitas kesehatan yang layak untuk cek kesehatan secara rutin (Lituwang, 2023). Diperlukan teknologi yang praktis dan efisien untuk memprediksi potensi diabetes masyarakat secara individu. 
Sebagian masyarakat tidak menyadari bahwa mereka menderita penyakit diabetes sampai gejala yang dimiliki semakin parah. Perihal membantu menciptakan teknologi yang mampu memprediksi diagnosis penyakit ini secara efisien, metode k-Nearest Neighbor (KNN) dapat diaplikasikan untuk memprediksi potensi seseorang terkena diabetes (Melinda et al., 2022). Algoritma KNN bekerja dengan mengelompokkan data berdasarkan tingkat kemiripan atau kedekatan dengan data lainnya (Sholeh et al., 2022). Selain algoritma KNN, terdapat algoritma lain, seperti Random Forest yang dapat digunakan untuk memprediksi diabetes seseorang (Yunita et al., 2021).

Referensi:

Liwutang, L. (2023). Analisis Perlindungan Hukum Terhadap Pasien Kurang Mampu Dalam Mendapatkan Pelayanan Kesehatan Yang Layak. Lex Privatum, 11(1), 115–123. Retrieved from https://ejournal.unsrat.ac.id/v3/index.php/lexprivatum/article/view/56925/46995

Melinda, K., Khasanah, S., & Susanto, A. (2022). Gambaran kadar gula darah penderita Diabetes Mellitus peserta Prolanis di Puskesmas 1 Sumbang Kabupaten Banyumas. Jurnal Inovasi Penelitian, 3(6), 6657–6670.

Sholeh, M., Andayati, D., & Rachmawati, R. Y. (2022). Data mining model klasifikasi menggunakan algoritma K-Nearest Neighbor dengan normalisasi untuk prediksi penyakit diabetes. TeIKa: Jurnal Teknologi Informatika dan Komputer, 12(2), 77–87. https://doi.org/10.36342/teika.v12i02.2911.

Yunita, R., & Zulfian, L. (2021). Penerapan Algoritma Random Forest Dalam Klasifikasi Penyakit Diabetes Mellitus. Jurnal Farmasi Klinik, 7(2), 105–110. Retrieved from https://e-journal.polkesraya.ac.id/index.php/jfk/article/download/229/100/424.


## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi potensi diabetes pada individu secara efisien dan akurat berdasarkan data kondisi kesehatan pasien menggunakan teknologi berbasis machine learning?

- Bagaimana perbandingan akurasi model dengan algoritma K-Nearest Neighbor (KNN) dan Random Forest dalam mengklasifikasikan risiko diabetes berdasarkan data kondisi kesehatan?

- Algoritma mana yang memiliki akurasi dan performa lebih baik dalam memprediksi diabetes?

### Goals

- Mampu membangun model machine learning yang mampu memprediksi potensi diabetes pada individu secara efisien dan akurat berdasarkan data kondisi kesehatan pasien.

- Membandingkan akurasi model dengan algoritma K-Nearest Neighbor (KNN) dan Random Forest dalam mengklasifikasikan risiko diabetes berdasarkan kondisi kesehatan.

- Menentukan model terbaik dalam memprediksi diabetes.

 ### Solution statements

- **Membangun model dengan algoritma K-Nearest Neighbor**

Algoritma KNN menggunakan pendekatan jarak untuk mengklasifikasikan seseorang ke dalam kategori berisiko atau tidak berisiko diabetes berdasarkan kemiripan data kondisi kesehatan yang dimiliki. Model ini akan diuji menggunakan nilai n_neighbor yang optimal. Evaluasi performa dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk mengukur seberapa baik performal model dalam memprediksi diabetes.

 - **Membangun model dengan algoritma Random Forest**

Algoritma Random Forest digunakan untuk memprediksi risiko diabetes dengan menggabungkan hasil dari sejumlah pohon keputusan. Model ini akan dievaluasi berdasarkan kombinasi parameter seperti n_estimators dan max_depth. Sana halnya dengan model KNN, evaluasi performa dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk mengukur seberapa baik performal model dalam memprediksi diabetes.

## Data Understanding
Data yang digunakan dalam penelitian ini berasal dari Kaggle dengan judul “Diabetes Prediction Dataset” yang tersedia di tautan:

https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

Dataset ini berisi beberapa kondisi kesehatan pasien/indikator yang dapat digunakan untuk memprediksi apakah individu memiliki diabetes.

### Variabel-variabel Prediksi Diabetes

**Variabel prediktor**

`gender`: jenis kelamin responden

`age`: usia responden (dalam tahun)

`hypertension`:  hipertensi (0 = tidak, 1 = ya)

`heart_disease`:  penyakit jantung (0 = tidak, 1 = ya)

`smoking_history`:  merokok responden

`bmi`: indeks massa tubuh responden

`HbA1c_level`: kadar HbA1c (hemoglobin terglikasi)

`blood_glucose_level`: kadar glukosa darah saat pemeriksaan

**Variabel Target**

`diabetes`: status diabetes responden (0 = tidak, 1 = ya) 

## Data Preprocessing
### Load Dataset

```
url = 'https://drive.google.com/file/d/1vadfxmry-PrWCsiwCIE9J5WcpxwGW8pJ/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
diabetes_df = pd.read_csv(path)
diabetes_df
```
RangeIndex: 100000 entries, 0 to 99999

Data columns (total 9 columns):

    Column               Non-Null Count   Dtype 

    0   gender               100000 non-null  object

    1   age                  100000 non-null  float64

    2   hypertension         100000 non-null  int64

    3   heart_disease        100000 non-null  int64  

    4   smoking_history      100000 non-null  object 

    5   bmi                  100000 non-null  float64

    6   HbA1c_level          100000 non-null  float64

    7   blood_glucose_level  100000 non-null  int64
 
    8   diabetes             100000 non-null  int64

### Deskripsi Variabel

- Dataset memiliki 3 kolom dengan tipe object, yaitu gender dan smoking_history. Kolom ini termasuk kategorik (fitur non-numerik).
- Terdapat 3 kolom numerik dengan tipe data float64 yaitu age, bmi, dan HbA1c_level.
- Terdapat 4 kolom numerik dengan tipe data int64, yaitu hypertension, heart_disease, blood_glucose_level, dan diabetes.

### Mengubah tipe data kolom heypertension, heart_disease, dan diabetes menjadi object

```
diabetes_df['hypertension'] = diabetes_df['hypertension'].astype('object')
diabetes_df['heart_disease'] = diabetes_df['heart_disease'].astype('object')
diabetes_df['diabetes'] = diabetes_df['diabetes'].astype('object')

```
```
diabetes_df.describe()
```
    age	bmi	HbA1c_level	blood_glucose_level

    count	100000.000000	100000.000000	100000.000000	100000.000000

    mean	41.885856	27.320767	5.527507	138.058060

    std	22.516840	6.636783	1.070672	40.708136

    min	0.080000	10.010000	3.500000	80.000000

    25%	24.000000	23.630000	4.800000	100.000000

    50%	43.000000	27.320000	5.800000	140.000000

    75%	60.000000	29.580000	6.200000	159.000000

    max	80.000000	95.690000	9.000000	300.000000

`age` (usia)
- Rata-rata usia pasien: 41.89 tahun dengan rentang 0.08 sampai 80 tahun.
- Sekitar 75% pasien berusia kurang dari atau sama dengan 60 tahun.
- Std dev: 22.52 menunjukkan penyebaran usia yang cukup luas.

`bmi` (Indeks Massa Tubuh)
- Mean: 27.32 menunjukkan rata-rata BMI di kategori overweight.
- Std dev: 6.64 menunjukkan variasi BMI cukup besar

`HbA1c_level` (Hemoglobin terglikasi)
- Rata-rata pasien memiliki Hemoglobin terglikasi 5.53.
- Sekitar 75% responden memiliki nilai HbA1c kurang dari atau sama dengan 6.2%.

`blood_glucose_level` (glukosa darah)
- Rata-rata pasien memiliki glukosa darah 138.06 mg/dL.
- Sekitar 75% responden memiliki glukosa darah 159 mg/dL.

#### Menangani Missing Value
Mengidentifikasi jumlah kolom yang tidak memiliki nilai

```
diabetes_df.isnull().sum()
```
    gender	0

    age	0

    hypertension	0

    heart_disease	0

    smoking_history	0

    bmi	0

    HbA1c_level	0

    blood_glucose_level	0

    diabetes	0

Mengidentifikasi kolom yang memiliki tipe data float64 yang memiliki nilai 0
```
age = (diabetes_df.age == 0).sum()
HbA1c_level = (diabetes_df.HbA1c_level == 0).sum()
blood_glucose_level = (diabetes_df.blood_glucose_level == 0).sum()

print("Nilai 0 di kolom age ada: ", age)
print("Nilai 0 di kolom HbA1c_level ada: ", HbA1c_level)
print("Nilai 0 di kolom blood_glucose_level ada: ", blood_glucose_level)
```
Nilai 0 di kolom age ada:  0

Nilai 0 di kolom HbA1c_level ada:  0

Nilai 0 di kolom blood_glucose_level ada:  0

Berdasarkan identifikasi, tidak ada missing value yang harus ditangani.

#### Menangani Outlier

**gender**
```
gender = diabetes_df['gender'].value_counts()
print(gender)
```
    gender

    Female    58552

    Male      41430

    Other        18

```
diabetes_df = diabetes_df[diabetes_df['gender'] != 'Other']
```

Berdasarkan identifikasi, terlihat bahwa kolom gender berisi 3 kategori, yaitu Female, Male, dan Other. Pada dasarnya gender terdiri dari 2 kategori, yaitu Female dan Male saja, sehingga 18 baris yang kolom Gender berisi Other sebaiknya dihapus agar tidak menimbulkan bias.

**age**

    age	   count

    80.00	5621

    51.00	1619

    47.00	1572

    48.00	1568

    49.00	1541

    ...	...

    0.48	83

    1.00	83

    0.40	66

    0.16	59

    0.08	36

Berdasarkan identifikasi, terlihat bahwa terdapat beberapa usia yang tidak sesuai, seperti dalam bentuk desimal. Oleh karena itu, perlu dibersihkan untuk mempermudah analisis dan tidak menimbulkan bias.

```
diabetes_df = diabetes_df[diabetes_df['age'] >= 1]
diabetes_df['age'] = diabetes_df['age'].astype('int64')
sns.boxplot(x=diabetes_df['age'])
```
![Boxplot age](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20112523.png?raw=true)

**hypertension dan heart_disease**
```
diabetes_df['hypertension'].unique()
diabetes_df['heart_disease'].unique()
```
array([0, 1], dtype=object)

array([1, 0], dtype=object)

Kolom hypertension tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki hipertensi) dan 1 (memiliki hipertensi).

Kolom heart_disease tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki penyakit jantung) dan 1 (memiliki penyakit jantung).

**smoking_history**
```
diabetes_df['smoking_history'].value_counts()
```
    count

    smoking_history	

    never	35055

    No Info	34946

    former	9352

    current	9285

    not current	6430

    ever	4003

Berdasarkan identifikasi, kolom smoking_history berisi 6 kategori, yaitu No Info, never, former, current, not current, dan ever. Untuk mempermudah analisis, sebaiknya dikategorikan lagi menjadi never (tidak pernah merokok), ever (pernah tapi sudah tidak merokok), dan current (saat ini perokok). Untuk former dan not current dapat dikategorikan ke ever karena sama-sama pernah merokok, tetapi saat ini sudah tidak merokok. Selanjutnya, kategori No Info sebanyak 35816 dipertahankan agar tidak menghilangkan banyak informasi karena frekuensinya yang sangat besar.

Mengubah kategori former, current, not current berubah menjadi ever
```
diabetes_df['smoking_history'] = diabetes_df['smoking_history'].replace(['former', 'not current'],'ever')
```

**bmi**
```
sns.boxplot(x=diabetes_df['bmi'])
```
![Boxplot BMI](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20112709.png?raw=true)

Menghapus baris dengan bmi > 60
```
diabetes_df = diabetes_df[diabetes_df['bmi'] < 60]
```

**HbA1c_level**
```
sns.boxplot(x=diabetes_df['HbA1c_level'])
```
![Boxplot HbA1c_level](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20112750.png?raw=true)
Terlihat bahwa kolom HbA1c_level memiliki banyak outliers, namun tidak perlu dihapus karena angka tersebut masih relevan secara medis.

**blood_glucose_level**
```
sns.boxplot(x=diabetes_df['blood_glucose_level'])
```
![Boxplot blood_glucose_level](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20112835.png?raw=true)

Terlihat bahwa kolom blood_glucose_level memiliki banyak outliers, namun tidak perlu dihapus karena angka tersebut masih relevan secara medis.

**diabetes**
```
diabetes_df['diabetes'].unique()
```
array([0, 1], dtype=object)

Output dari kode di atas menunjukkan bahwa kolom diabetes tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki penyakit diabetes) dan 1 (memiliki penyakit diabetes).

### Menangani duplikasi
```
diabetes_df.duplicated().sum()
diabetes_df.drop_duplicates(inplace=True)
```

### Univariate Analysis
Membagi fitur menjadi 2 bagian
```
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history', 'diabetes']
```
#### Categorical Features
**gender**
```
feature = categorical_features[0]
count = diabetes_df[feature].value_counts()
percent = 100*diabetes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
jumlah sampel  persentase

    gender        

    Female          55514        58.5

    Male            39386        41.5

Terdapat 2 kategori dalam fitur gender, yaitu Female dan Male. Sebesar 58.5% kategori pada fitur gender adalah Female yang artinya lebih banyak dari Male.

**hypertension**
```
feature = categorical_features[1]
count = diabetes_df[feature].value_counts()
percent = 100*diabetes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
jumlah sampel  persentase

    hypertension      

    0                     87463        92.2

    1                      7437         7.8

Terdapat 2 kategori dalam fitur hypertension, yaitu 0 (tidak memiliki  hipertensi) dan 1 (memiliki  hipertensi). Sebesar 92.2% pasien tidak memiliki  hipertensi dan 7.8% pasien memiliki  hipertensi.

**heart_disease**
```
feature = categorical_features[2]
count = diabetes_df[feature].value_counts()
percent = 100*diabetes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
 jumlah sampel  persentase

    heart_disease       

    0                      90986        95.9

    1                       3914         4.1

Terdapat 2 kategori dalam fitur heart_disease, yaitu 0 (tidak memiliki penyakit jantung) dan 1 (memiliki penyakit jantung). Sebesar 95.9% pasien tidak memiliki penyakit jantung dan 4.1% pasien memiliki penyakit jantung.

**smoking_history**
```
feature = categorical_features[3]
count = diabetes_df[feature].value_counts()
percent = 100*diabetes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
jumlah sampel  persentase

    smoking_history      

    never                    34305        36.1

    No Info                  31965        33.7

    ever                     19449        20.5

    current                   9181         9.7

Terdapat 4 kategori dalam fitur smoking_history, yaitu never (tidak pernah merokok), No Info (tidak memberi informasi), ever (pernah tapi sudah tidak merokok), dan current (saat ini perokok). Sebanyak 36.1% pasien tidak pernah merokok, sebanyak 33.7% pasien tidak memberikan informasi, sebanyak 20.5% pasien pernah tapi sudah berhenti merokok, dan 9.7% pasien saat ini perokok.

**diabetes**
```
feature = categorical_features[4]
count = diabetes_df[feature].value_counts()
percent = 100*diabetes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```
 jumlah sampel  persentase

    diabetes      

    0                 86456        91.1

    1                  8444         8.9

Terdapat 2 kategori dalam fitur diabetes, yaitu 0 (tidak memiliki penyakit diabetes) dan 1 (memiliki penyakit diabetes). Sebesar 91.1% pasien tidak memiliki penyakit diabetes dan 8.9% pasien memiliki penyakit diabetes.

##### Numerical Features
```
diabetes_df.hist(bins=50, figsize=(20,15))
plt.show()
```
![Histogram Fitur Numerik](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20113001.png?raw=true)

Histogram di atas menunjukkan bahwa:
- Distribusi usia cukup merata dari usia 0 sampai 80 tahun dan memperlihatkan bahwa sebagian besar pasien berada di usia produktif dan lansia. Terjadi lonjakan di usia 80 tahun.
- Distribusi bmi secara umum right-skewed, yang berarti banyak pasien yang memiliki BMI lebih tinggi dari normal. Terjadi lonjakan di sekitar angka 28.
- Distribusi HbA1c tidak berbentuk normal dan terjadi lonjakan di sekitar angka 6.
- blood_glucose_level tersebar dari sekitar 70 hingga hampir 300, tetapi dengan dominasi jumlah sampel pada angka-angka tertentu. Terjadi lonjakan besar pada nilai sekitar 155.

#### Multivariate Analysis
```
cat_features = diabetes_df.select_dtypes(include='object').columns.difference(['diabetes']).to_list()
for col in cat_features:
    sns.countplot(x=col, hue='diabetes', data=diabetes_df)
    plt.title(f'Distribusi Diabetes terhadap {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    ```
![alt text](image-6.png)

**gender**
- Mayoritas pasien memiliki gender Female.
- Perbandingan jumlah pasien yang tidak menderita diabetes pada Female dan Male cukup signifikan.
- Perbandingan jumlah pasien yang menderita diabetes pada Female dan Male hampir memiliki proporsi yang sama.

**heart_disease**
- Mayoritas pasien tidak memiliki penyakit jantung
- Perbandingan jumlah pasien yang tidak menderita diabetes pada pasien yang tidak memiliki penyakit jantung dan memiliki penyakit jantung cukup signifikan.
- Perbandingan jumlah pasien yang menderita diabetes pada pasien yang tidak memiliki penyakit jantung dan memiliki penyakit jantung hampir memiliki proporsi yang sama.

**hypertension**
- Perbandingan jumlah pasien yang tidak memiliki hipertensi dan memiliki hipertensi sangat signifikan.
- Perbandingan jumlah pasien yang tidak menderita diabetes pada pasien yang tidak memiliki hipertensi dan memiliki hipertensi cukup signifikan.
- Perbandingan jumlah pasien yang menderita diabetes pada pasien yang tidak memiliki hipertensi dan memiliki hipertensi hampir memiliki proporsi yang sama.

**smoking_history**
- Jumlah orang yang tidak merokok dan yang tidak memberi informasi riwayat merokok cukup besar. Jumlah penderita diabetes dalam kategori ini relatif lebih kecil secara proporsi dibanding total populasi kategori tersebut.
- Total pasien yang saat ini perokok lebih kecil dibanding kategori lain dan jumlah penderita diabetes pada kategori ini tidak berbeda signifikan/proporsi sebanding dengan total pasien yang saat ini perokok.
- Total pasien yang pernah tapi sudah tidak merokok lebih tinggi dibanding kategori current dan lebih rendah dibanding kategori lain. Jumlah penderita diabetes pada kategori ini tidak berbeda signifikan/proporsi sebanding dengan total pasien yang pernah tapi sudah tidak merokok.

```
sns.pairplot(diabetes_df[numerical_features + ['diabetes']], hue='diabetes', diag_kind='kde')
plt.show()

```
![Multivariate_num](https://github.com/sorayaindahs/predictive-analytics/blob/main/Screenshot%202025-05-23%20122429.png?raw=true)
- age: Distribusi usia pasien penderita diabetes cenderung lebih tinggi di usia dewasa-tua, dibandingkan dengan yang tidak diabetes yang tersebar lebih luas, termasuk usia muda.
- bmi: Distribusi bmi terlihat tidak jauh berbeda antara dua kelas, tetapi persebaran pasien yang memiliki diabetes sedikit lebih padat di rentang bmi > 25 (overweight/obesitas).
- HbA1c_level: Perbedaan sangat signifikan, pasien penderita diabetes dominan pada nilai HbA1c di atas 6.5, sedangkan non-diabetes lebih banyak di bawah 6.5. Hal ini relevan secara medis.
- blood_glucose_level: Perbedaan sangat signifikan, pasien penderita diabetes cenderung berada di nilai glukosa darah yang jauh lebih tinggi (≥ 200) dibandingkan yang non-diabetes.


- HbA1c_level vs blood_glucose_level: Terdapat hubungan positif walaupun tidak linier sempurna. Penderita diabetes cenderung memiliki HbA1c dan glukosa darah tinggi secara bersamaan. Titik oranye terkonsentrasi di kanan atas.

- age vs HbA1c_level/blood_glucose_level: Usia tidak terlalu berkorelasi langsung, tetapi penderita diabetes cenderung berada pada rentang usia pertengahan hingga lanjut.

- bmi vs fitur lainnya: Tidak ada korelasi kuat, tapi ada kecenderungan penderita diabetes muncul lebih banyak di BMI tinggi.

#### Numerical Features
```
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
correlations = {}
for col in numerical_features:
    corr, _ = pointbiserialr(diabetes_df[col], diabetes_df['diabetes'].astype(int))  # pastikan target dalam bentuk 0/1
    correlations[col] = round(corr, 3)
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Diabetes'])
plt.figure(figsize=(6, 4))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Korelasi Fitur Numerik dengan Target Kategorik (Diabetes)", fontsize=14)
plt.show()
```

Fitur `age` memiliki korelasi 0.26. Hal ini menunjukkan korelasi lemah positif. Usia lebih tua sedikit meningkatkan kemungkinan diabetes.

Fitur `bmi` memiliki korelasi 0.21. Hal ini menunjukkan korelasi lemah positif. BMI lebih tinggi sedikit berkaitan dengan diabetes.

Fitur `HbA1c_level` memiliki korelasi 0.41. Hal ini menunjukkan korelasi moderat positif. Kadar HbA1c tinggi cukup kuat berasosiasi dengan diabetes.

Fitur `blood_glucose_level` memiliki korelasi 0.42. Hal ini menunjukkan korelasi moderat positif. Kadar glukosa darah tinggi cukup kuat berkaitan dengan diabetes.

## Data Preparation

### Encoding Fitur Kategori

Proses encoding variabel kategorikal menggunakan LabelEncoder untuk mengubah data teks menjadi format numerik yang dapat diproses saat membuat model. Kolom seperti smoking_history, gender, hypertension, heart_disease, dan diabetes diubah nilainya menjadi angka sesuai urutan kategorinya.
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
diabetes_df['smoking_history'] = le.fit_transform(diabetes_df['smoking_history'])
diabetes_df['gender'] = le.fit_transform(diabetes_df['gender'])
diabetes_df['hypertension'] = le.fit_transform(diabetes_df['hypertension'])
diabetes_df['heart_disease'] = le.fit_transform(diabetes_df['heart_disease'])
diabetes_df['diabetes'] = le.fit_transform(diabetes_df['diabetes'])
```

### Train-Test_Split
Sebelum melakukan pemodelan, data dibagi menjadi data training sebanyak 80% dan data testing sebanyak 20% yang bertujuan agar model yang dilatih dapat memelajari pola pada data training dan menghasilkan performa yang baik.


```
X = diabetes_df.drop('diabetes', axis=1)
y = diabetes_df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Standarisasi
Standarisasi diterapkan pada fitur numerik menggunakan StandardScaler yang bertujuan untuk mengubah skala data agar memiliki rata-rata 0 dan standar deviasi 1. Hal ini dapat menyeimbangkan sakala antarfitur.


```
X = diabetes_df.drop('diabetes', axis=1)
y = diabetes_df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ['age', 'bmi', 'HbA1c_level','blood_glucose_level']
scaler = StandardScaler()
print(type(X_train)) 
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])

X_train[numerical_features].head()
```

## Model Development: K-Nearest Neighbor
Menyiapkan data frame untuk analisis ketiga model tersebut lebih dahulu.

```
models = pd.DataFrame(index=['train_acc', 'test_acc'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
```

Selanjutnya, melatih data dengan KNN. Sebelum membuat model, perlu diketahui nilai neighbor terbaik.


```
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')  # 5-fold CV
    cv_scores.append(scores.mean())

plt.plot(k_values, cv_scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Optimal k value')
plt.grid()
plt.show()

optimal_k = k_values[cv_scores.index(max(cv_scores))]
print(f"Nilai k terbaik: {optimal_k}")
```
![alt text](image-8.png)

Nilai k terbaik: 17

KNN menggunakan parameter n_neighbors=17 yang berarti model akan mempertimbangkan 17 tetangga terdekat dalam menentukan kelas suatu data, bertujuan untuk mencapai keseimbangan antara overfitting dan underfitting.
Selanjutnya, model dilatih menggunakan data train dan digunakan untuk memprediksi label pada data train dan data test.

## Model Development: K-Nearest Neighbor
```
rf_model = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)

rf_model.fit(X_train, y_train)

y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)
```
Random Forest Classifier dengan parameter n_estimators=50 yang berarti model membangun 50 decision trees untuk menghasilkan prediksi kolektif. Parameter max_depth=16 digunakan untuk membatasi depth maksimal.

## Evaluation

**K-Nearest Neighbors (KNN)**

```
models.loc['train_acc', 'KNN'] = accuracy_score(y_train, y_pred_train_knn)
models.loc['test_acc', 'KNN'] = accuracy_score(y_test, y_pred_test_knn)

print("KNN Classification Report (Test):")
print(classification_report(y_test, y_pred_test_knn))

print("KNN Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test_knn))
```
KNN Classification Report (Test):

              precision    recall  f1-score   support

           0       0.96      1.00      0.98     17270

           1       0.95      0.57      0.72      1710

    accuracy                           0.96     18980

   macro avg       0.96      0.79      0.85     18980

weighted avg       0.96      0.96      0.95     18980

KNN Confusion Matrix (Test):

[[17223    47]

 [  730   980]]


Kelebihan:

- Model KNN menunjukkan akurasi yang cukup tinggi, yaitu sebesar 96% yang berarti mampu mengklasifikasikan sebagian besar data secara tepat.

- Model mampu memprediksi pasien non-diabetes (kelas 0) sangat baik, ditunjukkan dengan nilai recall sempurna 1.00 yang artinya hampir tidak ada kesalahan dalam klasifikasi negatif.

- Nilai presisi untuk kelas positif (1) juga tinggi, yaitu 0.95 yang menandakan bahwa prediksi positif dari model ini umumnya benar.

Kelemahan:

- Meskipun niali presisi tinggi, recall untuk kelas 1 hanya mencapai 0.57 yang artinya model masih sering gagal mengenali pasien yang sebenarnya mengidap diabetes, masih terdapat 730 kasus false negative.

- Nilai F1-score kelas 1 cukup rendah, yaitu 0.72. Hal ini juga mencerminkan bahwa keseimbangan antara precision dan recall masih belum optimal.

**Random Forest**
```
models.loc['train_acc', 'RF'] = accuracy_score(y_train, y_pred_train_rf)
models.loc['test_acc', 'RF'] = accuracy_score(y_test, y_pred_test_rf)

print("Random Forest Classification Report (Test):")
print(classification_report(y_test, y_pred_test_rf))

print("Random Forest Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_pred_test_rf))
```
Random Forest Classification Report (Test):

              precision    recall  f1-score   support

           0       0.97      1.00      0.98     17270

           1       0.98      0.67      0.80      1710

    accuracy                           0.97     18980

   macro avg       0.97      0.84      0.89     18980

weighted avg       0.97      0.97      0.97     18980

Random Forest Confusion Matrix (Test):

[[17246    24]

 [  560  1150]]

Kelebihan:

Model Random Forest mempunyai akurasi lebih tinggi dibanding KNN, yaitu sebesar 97%. Recall kelas 1 sebesar 0.67 yang berarti lebih efektif dalam mengidentifikasi pasien yang menderita diabetes. Jumlah false negative juga lebih sedikit, yaitu 560.

Selain itu, F1-score untuk kelas 1 lebih tinggi, yaitu 0.80 yang mencerminkan keseimbangan yang lebih baik antara kemampuan memprediksi kelas positif.

Kelemahan:
Model ini masih menghasilkan sejumlah false negative yang cukup signifikan, sehingga belum sepenuhnya optimal untuk kebutuhan deteksi dini.

## Kesimpulan

Berdasarkan predictive analysis yang telah dilakukan, dapat disimpulkan bahwa metode machine learning dapat digunakan sebagai alat untuk memprediksi potensi seseorang mengidap diabetes secara efisien dan akurat. Menggunakan data pasien seperti usia, indeks massa tubuh (BMI), kadar glukosa darah, serta riwayat kesehatan lainnya. Algoritma klasifikasi yaitu K-Nearest Neighbor (KNN) dan Random Forest digunakan untuk membandingkan performa model dalam mengklasifikasikan data. 

Model KNN dan Random Forest sama-sama memiliki akurasi yang tinggi.  Meskipun demikian, berdasarkan evaluasi, model Random Forest menunjukkan performa yang lebih baik, terutama pada aspek akurasi, recall, dan F1-score, khususnya dalam mengidentifikasi pasien dengan diabetes. Keunggulan ini menjadikan Random Forest lebih layak untuk diterapkan dalam upaya deteksi dini penyakit ini.

Dapat disimpulkan bahwa Random Forest dapat menjadi model terbaik untuk memprediksi potensi diabetes karena mampu memberikan hasil yang lebih akurat.
