# Laporan Proyek Machine Learning - Soraya Indah Setiani

## Domain Proyek
Diabetes merupakan salah satu penyakit tidak menular yang sangat berdampak pada kondisi kesehatan masyarakat secara global. Menurut data yang dipublikasi oleh International Diabetes Federation (IDF), penderita diabetes di dunia pada tahun 2021 berada di angka 537 juta jiwa. Angka ini diprediksi akan terus meningkat akibat gaya hidup masyarakat. 
Prediksi untuk mengetahui potensi diabetes pada individu sangat penting untuk dilakukan guna mencegah penyakit diabetes secepat mungkin. Tidak semua masyarakat mampu mengakses fasilitas kesehatan yang layak untuk cek kesehatan secara rutin (Lituwang, 2023). Diperlukan teknologi yang praktis dan efisien untuk memprediksi potensi diabetes masyarakat secara individu. 
Sebagian masyarakat tidak menyadari bahwa mereka menderita penyakit diabetes sampai gejala yang dimiliki semakin parah. Perihal membantu menciptakan teknologi yang mampu memprediksi diagnosis penyakit ini secara efisien, metode K-Nearest Neighbor (KNN) dapat diaplikasikan untuk memprediksi potensi seseorang terkena diabetes (Melinda et al., 2022). Algoritma KNN bekerja dengan mengelompokkan data berdasarkan tingkat kemiripan atau kedekatan dengan data lainnya (Sholeh et al., 2022). Selain algoritma KNN, terdapat algoritma lain, seperti Random Forest yang dapat digunakan untuk memprediksi diabetes seseorang (Yunita et al., 2021).

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
- **Membangun model dengan algoritma K-Nearest Neighbor untuk Prediksi Diabetes**

Algoritma KNN menggunakan pendekatan jarak untuk mengklasifikasikan seseorang ke dalam kategori berisiko atau tidak berisiko diabetes berdasarkan kemiripan data kondisi kesehatan yang dimiliki. Model ini akan diuji menggunakan nilai n_neighbor yang optimal. Evaluasi performa dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk mengukur seberapa baik performa model dalam memprediksi diabetes.

- **Membangun model dengan algoritma Random Forest untuk Prediksi Diabetes**

Algoritma Random Forest digunakan untuk memprediksi risiko diabetes dengan menggabungkan hasil dari sejumlah pohon keputusan. Model ini akan dievaluasi berdasarkan kombinasi parameter seperti n_estimators dan max_depth. Sana halnya dengan model KNN, evaluasi performa dilakukan menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk mengukur seberapa baik performa model dalam memprediksi diabetes.

## Data Understanding
Data yang digunakan dalam penelitian ini berasal dari Kaggle dengan judul “Diabetes Prediction Dataset” yang tersedia di tautan:

https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data

Dataset ini berisi beberapa kondisi kesehatan pasien/indikator yang dapat digunakan untuk memprediksi apakah individu memiliki diabetes. Berikut tahapan selanjutnya untuk mengetahui informasi terkait dataset yang digunakan.

### Load Dataset

    RangeIndex: 100000 entries, 0 to 99999

    Data columns (total 9 columns):
    
    |   Column              | Non-Null Count     | Dtype    |
    |---------------------- |--------------------|----------|
    | 0  gender             | 100000 non-null    | object   |
    | 1  age                | 100000 non-null    | float64  |
    | 2  hypertension       | 100000 non-null    | int64    |
    | 3  heart_disease      | 100000 non-null    | int64    |
    | 4  smoking_history    | 100000 non-null    | object   |
    | 5  bmi                | 100000 non-null    | float64  |
    | 6  HbA1c_level        | 100000 non-null    | float64  |
    | 7  blood_glucose_level| 100000 non-null    | int64    |
    | 8  diabetes           | 100000 non-null    | int64    |

Diabetes Prediction Dataset memiliki **100000 baris** dan **9 kolom**.

- Dataset memiliki 3 kolom dengan tipe object, yaitu gender dan smoking_history. Kolom ini termasuk kategorik (fitur non-numerik).
- Terdapat 3 kolom numerik dengan tipe data float64 yaitu age, bmi, dan HbA1c_level.
- Terdapat 4 kolom numerik dengan tipe data int64, yaitu hypertension, heart_disease, blood_glucose_level, dan diabetes.

### Deskripsi Variabel
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

### Cek Summary
Summary dilakukan pada data numerik saja, sehingga kolom heypertension, heart_disease, dan diabetes yang masih bertipe float64 perlu diubah ke object terlebih dahulu karena merupakan kolom yang berisi kategori (0=tidak dan 1=ya).

    |        | age           | bmi           | HbA1c_level    | blood_glucose_level  |
    |--------|-------------  |-------------  |-------------   |----------------------|
    | count  | 100000.000000 | 100000.000000 | 100000.000000  | 100000.000000        |
    | mean   | 41.885856     | 27.320767     | 5.527507       | 138.058060           |
    | std    | 22.516840     | 6.636783      | 1.070672       | 40.708136            |
    | min    | 0.080000      | 10.010000     | 3.500000       | 80.000000            |
    | 25%    | 24.000000     | 23.630000     | 4.800000       | 100.000000           |
    | 50%    | 43.000000     | 27.320000     | 5.800000       | 140.000000           |
    | 75%    | 60.000000     | 29.580000     | 6.200000       | 159.000000           |
    | max    | 80.000000     | 95.690000     | 9.000000       | 300.000000           |


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

### Identifikasi Missing Value
**Identifikasi missing value setiap kolom**

    gender	0

    age	0

    hypertension	0

    heart_disease	0

    smoking_history	0

    bmi	0

    HbA1c_level	0

    blood_glucose_level	0

    diabetes	0

Secara keseluruhan, tidak terdapat kolom yang memiliki missing value, namun perlu diidentifikasi lebih lanjut terutama pada kolom yang memiliki tipe data float64

**Mengidentifikasi kolom age, HbA1c_level, dan blood_glucose_level (float64) yang memiliki nilai 0**

Kolom age, HbA1c_level, dan blood_glucose_level memiliki tipe data float64 dan tidak mungkin bernilai 0. Apabila terdapat nilai 0 pada 3 fitur tersebut, maka tidak relevan.

    Nilai 0 di kolom age ada:  0

    Nilai 0 di kolom HbA1c_level ada:  0

    Nilai 0 di kolom blood_glucose_level ada:  0

Berdasarkan identifikasi, tidak ada missing valuepada kolom age, HbA1c_level, dan blood_glucose_level yang harus ditangani.

**Identifikasi Duplikasi Data**
    
    np.int64(3854)

Kode tersebut menunjukkan bahwa dataset memiliki duplikasi sebanyak 3854 baris yang harus dibersihkan.

### Univariate Exploratory Data Analysis
Pada tahap ini akan mengidentifikasi missing value dan outlier atau unique pada setiap fitur.

**Fitur gender**


    | gender | jumlah sampel  | persentase (%) |
    |--------|----------------|----------------|
    | Female | 58552          | 58.6           |
    | Male   | 41430          | 41.4           |
    | Other  | 18             | 0.0            |


Berdasarkan identifikasi, terlihat bahwa kolom gender berisi 3 kategori, yaitu Female, Male, dan Other. Pada dasarnya gender terdiri dari 2 kategori, yaitu Female dan Male saja, sehingga 18 baris yang kolom Gender berisi Other sebaiknya dihapus agar tidak menimbulkan bias.

**Fitur age**

        | age   | count |
        |-------|-------|
        | 80.00 | 5621  |
        | 51.00 | 1619  |
        | 47.00 | 1572  |
        | 48.00 | 1568  |
        | 49.00 | 1541  |
        | ...   | ...   |
        | 0.48  | 83    |
        | 1.00  | 83    |
        | 0.40  | 66    |
        | 0.16  | 59    |
        | 0.08  | 36    |

Berdasarkan identifikasi, terlihat bahwa terdapat beberapa usia yang tidak sesuai, seperti dalam bentuk desimal. Oleh karena itu, perlu dibersihkan untuk mempermudah analisis dan tidak menimbulkan bias.

**Fitur hypertension**

                      jumlah sampel  persentase

    hypertension      

    0                     92515        92.5

    1                      7485        7.5

Kolom hypertension tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki hipertensi) dan 1 (memiliki hipertensi).

Terdapat 2 kategori dalam fitur hypertension, yaitu 0 (tidak memiliki  hipertensi) dan 1 (memiliki  hipertensi). Sebesar 92.5% pasien tidak memiliki  hipertensi dan 7.5% pasien memiliki  hipertensi.

**Fitur heart_disease**

    jumlah sampel  persentase

    heart_disease       

    0                      96058        96.1

    1                       3942         3.9

Kolom heart_disease tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki penyakit jantung) dan 1 (memiliki penyakit jantung).

Terdapat 2 kategori dalam fitur heart_disease, yaitu 0 (tidak memiliki penyakit jantung) dan 1 (memiliki penyakit jantung). Sebesar  96.1% pasien tidak memiliki penyakit jantung dan 3.9% pasien memiliki penyakit jantung.


**Fitur smoking_history**

    **Fitur smoking_history**

    | smoking_history  | count  | persentase |
    |------------------|--------|------------|
    | No Info          | 35816  | 35.8       |
    | Never            | 35095  | 35.1       |
    | former           | 9352   | 9.4        |
    | current          | 9286   | 9.3        |
    | not current      | 6447   | 6.4        |
    | ever             | 4004   | 4.0        |

Berdasarkan identifikasi, kolom smoking_history berisi 6 kategori, yaitu No Info, never, former, current, not current, dan ever. Untuk mempermudah analisis, sebaiknya dikategorikan lagi menjadi never (tidak pernah merokok), ever (pernah tapi sudah tidak merokok), dan current (saat ini perokok). Untuk former dan not current dapat dikategorikan ke ever karena sama-sama pernah merokok, tetapi saat ini sudah tidak merokok. Selanjutnya, kategori No Info sebanyak  35816 dipertahankan agar tidak menghilangkan banyak informasi karena frekuensinya yang sangat besar.

**Fitur bmi**

![image](https://github.com/user-attachments/assets/caa8e6db-e8a8-4f58-af65-db484c8e02e3)

bmi > 60 perlu dihapus karena di luar batas maksimal dan tidak relevan secara medis pada umumnya.

**Fitur HbA1c_level**

![image](https://github.com/user-attachments/assets/c587af19-0a31-4f04-aeaf-5d182c06d62f)

Terlihat bahwa kolom HbA1c_level memiliki banyak outliers, namun tidak perlu dihapus karena angka tersebut masih relevan secara medis.

**Fitur blood_glucose_level**

![image](https://github.com/user-attachments/assets/a1934655-bbb0-4cb4-b729-d6c9f0aee52d)


Terlihat bahwa kolom blood_glucose_level memiliki banyak outliers, namun tidak perlu dihapus karena angka tersebut masih relevan secara medis.

**Fitur diabetes**

                  jumlah sampel  persentase

    diabetes      

    0                 91500        91.5

    1                  8500         8.5

Output di atas menunjukkan bahwa kolom diabetes tidak memiliki outlier karena kategori sesuai, yaitu 0 (tidak memiliki penyakit diabetes) dan 1 (memiliki penyakit diabetes).

Terdapat 2 kategori dalam fitur diabetes, yaitu 0 (tidak memiliki penyakit diabetes) dan 1 (memiliki penyakit diabetes). Sebesar 91.5% pasien tidak memiliki penyakit diabetes dan 8.5% pasien memiliki penyakit diabetes.

**Histogram Fitur Numerik (age, bmi, HbA1c_level, dan blood_glucose_level)**

![image](https://github.com/user-attachments/assets/50492595-c791-4163-8fc1-98a2c6055e24)


Histogram di atas menunjukkan bahwa:
- Distribusi usia cukup merata dari usia 0 sampai 80 tahun dan memperlihatkan bahwa sebagian besar pasien berada di usia produktif dan lansia. Terjadi lonjakan di usia 80 tahun.
- Distribusi bmi secara umum right-skewed, yang berarti banyak pasien yang memiliki BMI lebih tinggi dari normal. Terjadi lonjakan di sekitar angka 28.
- Distribusi HbA1c tidak berbentuk normal dan terjadi lonjakan di sekitar angka 6.
- blood_glucose_level tersebar dari sekitar 70 hingga hampir 300, tetapi dengan dominasi jumlah sampel pada angka-angka tertentu. Terjadi lonjakan besar pada nilai sekitar 155.

### Multivariate Exploratory Data Analysis
**Berikut merupakan Histogram Fitur Kategorik.**

![image](https://github.com/user-attachments/assets/ec38678c-1b44-448b-bf07-b14ccf0d102b)


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

**Berikut merupakan Histogram Fitur numerik.**

![image](https://github.com/user-attachments/assets/06671e36-ea8a-4410-b719-d513d82e9b5a)

- age: Distribusi usia pasien penderita diabetes cenderung lebih tinggi di usia dewasa-tua, dibandingkan dengan yang tidak diabetes yang tersebar lebih luas, termasuk usia muda.
- bmi: Distribusi bmi terlihat tidak jauh berbeda antara dua kelas, tetapi persebaran pasien yang memiliki diabetes sedikit lebih padat di rentang bmi > 25 (overweight/obesitas).
- HbA1c_level: Perbedaan sangat signifikan, pasien penderita diabetes dominan pada nilai HbA1c di atas 6.5, sedangkan non-diabetes lebih banyak di bawah 6.5. Hal ini relevan secara medis.
- blood_glucose_level: Perbedaan sangat signifikan, pasien penderita diabetes cenderung berada di nilai glukosa darah yang jauh lebih tinggi (≥ 200) dibandingkan yang non-diabetes.


- HbA1c_level vs blood_glucose_level: Terdapat hubungan positif walaupun tidak linier sempurna. Penderita diabetes cenderung memiliki HbA1c dan glukosa darah tinggi secara bersamaan. Titik oranye terkonsentrasi di kanan atas.

- age vs HbA1c_level/blood_glucose_level: Usia tidak terlalu berkorelasi langsung, tetapi penderita diabetes cenderung berada pada rentang usia pertengahan hingga lanjut.

- bmi vs fitur lainnya: Tidak ada korelasi kuat, tapi ada kecenderungan penderita diabetes muncul lebih banyak di BMI tinggi.

## Data preparation
### Membersihkan Missing Value, Outlier, atau Data yang Tidak Relevan
1) Berdasarkan exploratory data pada fitur gender yang telah dilakukan, berikut jumlah data pada fitur gender setelah kategori Other dihapus.

                   jumlah sampel  persentase

        gender        

       Female          55514        58.5

       Male            39386        41.5

2) Berdasarkan exploratory data pada fitur age yang telah dilakukan, langkah yang perlu dilakukan adalah menghapus nilai age yang tidak sesuai, yaity  < 1. Setelah itu, tipe data diubah menjadi int64 agar lebih sesuai dan mudah dianalisis. Berikut boxplot age setelah menghapus beberapa nilai yang tidak sesuai.

![image](https://github.com/user-attachments/assets/5b068d50-89b0-4bb9-9eef-0ee6acb656f7)


3) Berdasarkan exploratory data pada fitur age yang telah dilakukan, berikut jumlah data pada fitur age setelah kategori bmi > 60 dihapus.

![image](https://github.com/user-attachments/assets/ee74782e-f6f3-4da3-97c9-90bb951d4796)


### Mengubah Kategori

Mengubah kategori pada fitur smoking_history yang berkategori former, current, not current berubah menjadi ever

jumlah sampel  persentase

    smoking_history      

    | smoking_history  | count  | percentage (%) |
    |------------------|--------|----------------|
    | never            | 35002  | 35.4           |
    | No Info          | 34917  | 35.3           |
    | ever             | 19766  | 20.0           |
    | current          | 9270   | 9.4            |


Terdapat 4 kategori dalam fitur smoking_history, yaitu never (tidak pernah merokok), No Info (tidak memberi informasi), ever (pernah tapi sudah tidak merokok), dan current (saat ini perokok). Sebanyak 35.4% pasien tidak pernah merokok, sebanyak 35.3% pasien tidak memberikan informasi, sebanyak 20.0% pasien pernah tapi sudah berhenti merokok, dan 9.4% pasien saat ini perokok.

### Menghapus Duplikasi Data
Berdasarkan identifikasi duplikasi data, terdapat duplikasi sebanyak 3854 baris yang harus dibersihkan. Setelah dibersihkan jumlah baris menjadi sebanyak 94900 baris

### Cek Korelasi

![image](https://github.com/user-attachments/assets/f40fe124-413c-4671-93df-29220d76dc1a)


Fitur `age` memiliki korelasi 0.26. Hal ini menunjukkan korelasi lemah positif. Usia lebih tua sedikit meningkatkan kemungkinan diabetes.

Fitur `bmi` memiliki korelasi 0.21. Hal ini menunjukkan korelasi lemah positif. BMI lebih tinggi sedikit berkaitan dengan diabetes.

Fitur `HbA1c_level` memiliki korelasi 0.41. Hal ini menunjukkan korelasi moderat positif. Kadar HbA1c tinggi cukup kuat berasosiasi dengan diabetes.

Fitur `blood_glucose_level` memiliki korelasi 0.42. Hal ini menunjukkan korelasi moderat positif. Kadar glukosa darah tinggi cukup kuat berkaitan dengan diabetes.

### Encoding Fitur Kategori
Proses encoding variabel kategorikal menggunakan LabelEncoder untuk mengubah data teks menjadi format numerik yang dapat diproses saat membuat model. Kolom seperti smoking_history, gender, hypertension, heart_disease, dan diabetes diubah nilainya menjadi angka sesuai urutan kategorinya.

    | Index | gender  | age | hypertension | heart_disease  | smoking_history  | ... | diabetes |
    |--------|--------|-----|--------------|----------------|------------------|-----|----------|
    | 0      | 0      | 80  | 0            | 1              | 3                | ... | 0        |
    | 1      | 0      | 54  | 0            | 0              | 0                | ... | 0        |
    | 2      | 1      | 28  | 0            | 0              | 3                | ... | 0        |
    | 3      | 0      | 36  | 0            | 0              | 1                | ... | 0        |
    | 4      | 1      | 76  | 1            | 1              | 1                | ... | 0        |
    | ...    | ...    | ... | ...          | ...            | ...              | ... | ...      |
    | 99994  | 0      | 36  | 0            | 0              | 0                | ... | 0        |
    | 99996  | 0      | 2   | 0            | 0              | 0                | ... | 0        |
    | 99997  | 1      | 66  | 0            | 0              | 2                | ... | 0        |
    | 99998  | 0      | 24  | 0            | 0              | 3                | ... | 0        |
    | 99999  | 0      | 57  | 0            | 0              | 1                | ... | 0        |

### Train-Test-Split
Sebelum melakukan pemodelan, data dibagi menjadi data training sebanyak 80% dan data testing sebanyak 20% yang bertujuan agar model yang dilatih dapat memelajari pola pada data training dan menghasilkan performa yang baik.

### Standarisasi
Standarisasi diterapkan pada fitur numerik menggunakan StandardScaler yang bertujuan untuk mengubah skala data agar memiliki rata-rata 0 dan standar deviasi 1. Hal ini dapat menyeimbangkan skala antarfitur.

    | Statistik | age         | bmi         | HbA1c_level | blood_glucose_level  |
    |-----------|-------------|-------------|-------------|----------------------|
    | count     | 7.592e+04   | 7.592e+04   | 7.592e+04   | 7.592e+04            |
    | mean      | 1.4319e-17  | 1.9780e-16  | -1.0051e-15 | -1.3739e-16          |
    | std       | 1.000007    | 1.000007    | 1.000007    | 1.000007             |
    | min       | -1.851772   | -2.635134   | -1.894611   | -1.420121            |
    | 25%       | -0.816930   | -0.585842   | -0.683438   | -0.932065            |
    | 50%       | 0.037940    | -0.007486   | 0.248233    | 0.044048             |
    | 75%       | 0.802824    | 0.387193    | 0.620902    | 0.507701             |
    | max       | 1.702687    | 4.951802    | 3.229581    | 3.948496             |


- Nilai mean untuk fitur numerik (age, bmi, HbA1c_level, blood_glucose_level) sangat mendekati 0.
- Nilai st deviasi semua fitur tersebut adalah 1.000007, yang sangat dekat dengan 1.
- Ketidaksempurnaan (tidak bulat 0 dan 1) disebabkan oleh beberapa presisi dalam komputasi.

## Model Development: K-Nearest Neighbor
Sebelum membangun model, perlu meenyiapkan data frame untuk analisis kedua model tersebut lebih dahulu.
```
models = pd.DataFrame(index=['train_acc', 'test_acc'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
```
Selanjutnya, melatih data dengan KNN. Sebelum membuat model, perlu diketahui nilai neighbor terbaik.

![image](https://github.com/user-attachments/assets/a14df036-9ac6-4867-9581-76298e1c7dbe)


Nilai k terbaik: 17

Karena nilai k terbaik adalah 17, maka menggunakan parameter n_neighbors=17.
Selanjutnya, model dilatih menggunakan data train dan digunakan untuk memprediksi label pada data train dan data test.

```
# Inisialisasi model KNN untuk klasifikasi
knn_model = KNeighborsClassifier(n_neighbors=17)

# Melatih model KNN
y_train = y_train.astype(int)
y_test = y_test.astype(int)
knn_model.fit(X_train, y_train)

# Prediksi
y_pred_train_knn = knn_model.predict(X_train)
y_pred_test_knn = knn_model.predict(X_test)
```
**Parameter model KNN:**

- Pada analisis ini, KNN menggunakan parameter n_neighbors=17 yang berarti model akan mempertimbangkan 17 tetangga terdekat dalam menentukan kelas suatu data. Label yang paling banyak muncul di antara 17 tetangga akan dijadikan hasil prediksi.
- y_train = y_train.astype(int) dan y_test = y_test.astype(int) untuk memastikan bahwa label target berupa angka integer karena KNN membutuhkan label yang bisa dihitung frekuensinya.
- knn_model.fit(X_train, y_train) untuk melatih model dengan menyimpan seluruh data latih dan akan mencari tetangga saat prediksi dilakukan.
- y_pred_train_knn = knn_model.predict(X_train)
Untuk memprediksi hasil pada data latih.
- y_pred_test_knn = knn_model.predict(X_test) untuk memprediksi hasil pada data uji/test.

**Cara kerja KNN:**

- KNN mengklasifikasikan suatu data uji dengan melihat tetangga terdekatnya di data latih.
- Menghitung jarak antara data uji dengan seluruh data latih.
- Mengambil 17 tetangga terdekat karena n_neighbors=17.
- Menentukan label berdasarkan mayoritas tetangga tersebut.

**Kelebihan model KNN:**

- Mudah melakukan klasifikasi, cukup dengan melihat tetangga terdekat.
- Tidak memerlukan pelatihan model kompleks.

**Kekurangan model KNN:**

Curse of Dimensionality, yaitu saat jumlah fitur (dimensi) bertambah banyak, semua data cenderung memiliki jarak sama, sehingga konsep tetangga terdekat sulit dianalisis.

## Model Development: Random Forest

```
# Inisialisasi model Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)

# Training model
rf_model.fit(X_train, y_train)

# Prediksi khusus untuk Random Forest
y_pred_train_rf = rf_model.predict(X_train)
y_pred_test_rf = rf_model.predict(X_test)
```
**Parameter model Random Forest:**

- n_estimator: jumlah trees di forest. Pada analisis ini menggunakan n_estimators=50 yang berarti jumlah pohon keputusan (decision trees) yang dibentuk dalam ensemble Random Forest sebanyak 50.
- max_depth: kedalaman atau panjang pohon yang merupakan ukuran seberapa banyak pohon dapat melakukan splitting untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. max_depth=16 artinya setiap pohon boleh bercabang hingga 16 level 
- random_state: digunakan untuk mengontrol angka acak generator yang digunakan. random_state=55, angka 55 tidak punya makna khusus, tetapi merupakan nilai tetap.
- n_jobs: jumlah job yang digunakan secara paralel yang merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.
- predict(X_train): untuk melihat performa model di data training.
- predict(X_test): untuk menguji generalisasi model ke data yang belum pernah dilihat.

**Cara kerja model Random Forest:**

- Model membuat 50 pohon keputusan karena n_estimators=50.
- Setiap pohon dilatih pada subset acak dari data latih.
- Kedalaman maksimum pohon dibatasi sampai 16 level agar tidak terlalu kompleks dan menghindari overfitting.
- Setelah semua pohon terbentuk, prediksi akhir dilakukan dengan melihat dari mayoritas pohon yang terbentuk.

**Kelebihan model Random Forest:**

- Akurasi tinggi
- Adanya pohon-pohon yang terbentuk membuat model lebih stabil dan tidak mudah overfitting.

**Kekurangan model Random Forest:**

- Kurang interpretatif karena terlalu banyak pohon, sehingga sulit dipahami.
- Membutuhkan waktu lebih lama karena training banyak pohon.


## Evaluasi

Untuk mengevaluasi model KNN dan Random Forest dalam melakukan prediksi diabetes, diperlukan evaluasi menggunakan metrik sebagai berikut.

- Accuracy: proporsi prediksi benar.
- Precision: Proporsi data yang diprediksi positif (menderita diabetes) dan true positif.
- Recall (Sensitivity): Proporsi data positif aktual (menderita diabetes)sebenarnya yang berhasil dikenali oleh model.
- F1-Score: Keseimbangan antara ketepatan (precision) dan kelengkapan deteksi (recall).
- Confusion Matrix: Matriks untuk melihat performa klasifikasi per kelas (true positive, false positive, true negative, dan false negative).

**K-Nearest Neighbors (KNN)**
KNN Classification Report (Test):

              precision    recall  f1-score   support

        0         0.96      1.00      0.98     17270

        1         0.95      0.57      0.72      1710

    accuracy                           0.96     18980

     macro avg    0.96      0.79      0.85      18980

    weighted avg  0.96      0.96      0.95      18980

    KNN Confusion Matrix (Test):

    [[17223    47]

    [  730   980]]

Berdasarkan output evaluasi menggunakan beberapa metrik untuk model KNN, dapat disimpulkan bahwa:

- Menunjukkan akurasi yang cukup tinggi, yaitu sebesar 96% yang berarti mampu mengklasifikasikan sebagian besar data secara tepat.

- Model mampu memprediksi pasien non-diabetes (kelas 0) sangat baik, ditunjukkan dengan nilai recall sempurna 1.00 yang artinya hampir tidak ada kesalahan dalam klasifikasi negatif.

- Nilai presisi untuk kelas positif (1) juga tinggi, yaitu 95% yang menandakan bahwa prediksi positif dari model ini umumnya benar.

- Meskipun nilai presisi tinggi, recall untuk kelas 1 hanya mencapai 57% yang artinya model masih sering gagal mengenali pasien yang sebenarnya mengidap diabetes, masih terdapat 730 kasus false negative.

- Nilai F1-score kelas 1 cukup rendah, yaitu 72%. Hal ini juga mencerminkan bahwa keseimbangan antara precision dan recall masih belum optimal.

**Random Forest**
    Random Forest Classification Report (Test):

              precision    recall  f1-score   support

           0       0.97      1.00      0.98     17270

           1       0.98      0.67      0.80      1710

    accuracy                           0.97     18980

    macro avg       0.97      0.84     0.89     18980

    weighted avg     0.97     0.97     0.97     18980

    Random Forest Confusion Matrix (Test):

    [[17246    24]

    [  560  1150]]

Berdasarkan output evaluasi menggunakan beberapa metrik untuk model Random Forest, dapat disimpulkan bahwa:

- Model Random Forest mempunyai akurasi lebih tinggi dibanding KNN, yaitu sebesar 97%. Recall kelas 1 sebesar 67% yang berarti lebih efektif dalam mengidentifikasi pasien yang menderita diabetes. Jumlah false negative juga lebih sedikit, yaitu 560.

- Selain itu, F1-score untuk kelas 1 lebih tinggi, yaitu 80% yang mencerminkan keseimbangan yang lebih baik antara kemampuan memprediksi kelas positif.

- Model ini masih menghasilkan sejumlah false negative yang cukup signifikan.

- Dapat disimpulkan bahwa Random Forest memiliki performa lebih baik secara keseluruhan untuk melakukan prediksi penyakit diabetes dibandingkan dengan model KNN.

**Keterkaitan dengan Business Understanding**

*Bagaimana memprediksi potensi diabetes secara efisien dan akurat?*

Model KNN dan Random Forest telah dibangun menggunakan data pasien berisi
age, BMI, HbA1c_level, blood_glucose_level, hypertension, dan smoking_history. Model ini mampu melakukan klasifikasi apakah seorang pasien berisiko menderita diabetes secara cepat dan akurat.

Oleh karena itu, model yang dibangun telah mencapai goal, yaitu mampu membangun model machine learning yang mampu memprediksi potensi diabetes pada individu secara efisien dan akurat berdasarkan data kondisi kesehatan pasien.

*Bagaimana membandingkan performa KNN vs Random Forest?*

Perbandingan performa dilakukan melalui evaluasi dengan metrik klasifikasi (accuracy, precision, recall, dan f1-score.) dan confusion matrix.
 
Oleh karena itu, analisis yang dilakukan telah mencapai goal untuk membandingkan akurasi model dengan algoritma K-Nearest Neighbor (KNN) dan Random Forest dalam mengklasifikasikan risiko diabetes berdasarkan kondisi kesehatan.

*Algoritma mana yang paling optimal untuk kasus ini?*

Random Forest dipilih karena memiliki:
- Accuracy: Random Forest 97%, sedangkan KNN  96%.
- Recall (kelas diabetes): Random Forest 67%, sedangkan KNN 57%.
- F1-score (kelas diabetes): Random Forest 0.80, sedangkan KNN 0.72.

Dapat disimpulkan bahwa Random Forest memiliki performa lebih baik secara keseluruhan untuk melakukan prediksi penyakit diabetes dibandingkan dengan model KNN. Oleh karena itu, analisis ini telah mencapai goal dalam menentukan model terbaik untuk memprediksi diabetes.

Hasil pemodelan ini tentu saja akan berdampak pada masyarakat apabila diimplementasikan dengan baik, contohnya menerapkan pada website, aplikasi, atau dashboard kesehatan yang dapat diakses secara mudah oleh masyarakat. Hal ini akan meningkatkan deteksi dini masyarakat terhadap penyakit diabetes, sehingga keparahan gejala diabetes dapat dicegah secepatnya.

