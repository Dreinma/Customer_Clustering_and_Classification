# Proyek Segmentasi dan Klasifikasi Pelanggan Transaksional

## Latar Belakang

Dalam dunia bisnis ritel, pemahaman terhadap perilaku pelanggan menjadi salah satu faktor krusial dalam menentukan strategi pemasaran, peningkatan layanan, serta pengambilan keputusan berbasis data.

Proyek ini menggunakan pendekatan machine learning untuk mencapai dua tujuan utama:
1.  **Segmentasi Pelanggan (Clustering):** Menggunakan metode unsupervised learning untuk mengelompokkan pelanggan ke dalam beberapa segmen berdasarkan karakteristik dan perilaku transaksi mereka.
2.  **Prediksi Segmen (Klasifikasi):** Membangun model supervised learning yang dapat memprediksi segmen seorang pelanggan baru berdasarkan datanya, memanfaatkan hasil dari tahap clustering.

## 📜 Dataset

Dataset yang digunakan adalah **"Financial Transactions"** yang berisi data transaksional fiktif pelanggan. Dataset ini dimuat langsung melalui URL Google Drive di dalam notebook. Fitur-fitur utamanya meliputi informasi transaksi (`TransactionAmount`, `TransactionType`, `Channel`), demografi (`CustomerAge`, `CustomerOccupation`), dan lokasi.

## ⚙️ Alur Kerja Proyek

Proyek ini dibagi menjadi dua tahap utama yang dijalankan secara berurutan:

### **Tahap 1: Segmentasi Pelanggan (Clustering.ipynb)**

Tujuan dari tahap ini adalah untuk menemukan kelompok-kelompok alami di dalam data pelanggan tanpa label yang sudah ada.

#### **1. Pra-pemrosesan dan Rekayasa Fitur (Feature Engineering)**
Data mentah dibersihkan dan diperkaya melalui beberapa langkah kunci:
* **Imputasi Data Hilang:** Nilai yang hilang pada fitur numerik diisi dengan **median**, sedangkan pada fitur kategorikal diisi dengan **modus**.
* **Penghapusan Data Duplikat:** Baris data yang identik sepenuhnya dihapus untuk memastikan kualitas model.
* **Penanganan Outlier:** Outlier pada fitur `TransactionAmount` ditangani dengan metode **capping** berdasarkan rentang IQR (Interquartile Range) untuk mengurangi efek nilai ekstrem tanpa kehilangan data.
* **Rekayasa Fitur:**
    * Fitur waktu seperti `TransactionHour`, `DayOfWeek`, dan `IsWeekend` diekstrak dari `TransactionDate`.
    * Fitur rasio `TransactionToBalanceRatio` dibuat untuk mengukur proporsi transaksi terhadap saldo akun.
    * Fitur `CustomerAge` dikelompokkan (binning) menjadi kategori `Muda`, `Dewasa`, dan `Lansia` untuk menangkap pola perilaku berbasis kelompok usia.
* **Transformasi Data:**
    * Fitur numerik di-scaling menggunakan `StandardScaler`.
    * Fitur kategorikal di-encode menggunakan `LabelEncoder`.

#### **2. Pembangunan Model Clustering**
* **Algoritma:** Model **K-Means** digunakan untuk melakukan segmentasi.
* **Penentuan Jumlah Cluster (k):** Jumlah cluster optimal ditentukan menggunakan `KElbowVisualizer` dari library `yellowbrick` dengan metrik `silhouette score` untuk mendapatkan hasil yang lebih andal secara statistik.
* **Hasil:** Model menemukan bahwa **k=2** adalah jumlah cluster yang paling optimal.

#### **3. Hasil Segmentasi**
Analisis mendalam pada kedua cluster menunjukkan bahwa pembeda utama antar segmen bukanlah perilaku finansial, melainkan **lokasi geografis**.
* **Cluster 0: Pelajar Pengguna Cabang - Fort Worth:** Segmen ini didominasi oleh pelajar muda yang melakukan transaksi debit di kantor cabang (branch) yang berlokasi di Fort Worth.
* **Cluster 1: Pelajar Pengguna Cabang - Oklahoma City:** Segmen ini memiliki profil yang nyaris identik dengan Cluster 0, namun lokasinya terpusat di Oklahoma City.

---

### **Tahap 2: Prediksi Segmen (Klasifikasi.ipynb)**

Tujuan dari tahap ini adalah untuk melatih model yang dapat secara otomatis mengklasifikasikan pelanggan baru ke dalam segmen yang sudah ditemukan.

#### **1. Persiapan Data**
* **Input:** Menggunakan file `data_clustering.csv` yang merupakan output dari notebook clustering. File ini berisi data yang sudah sepenuhnya diproses (scaled dan encoded).
* **Target (Y):** Kolom `Target` (hasil dari label cluster 0 dan 1) menjadi variabel yang akan diprediksi.
* **Fitur (X):** Semua kolom lainnya menjadi fitur prediktor.
* **Data Splitting:** Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan `train_test_split` dengan parameter `stratify=y` untuk menjaga proporsi kelas yang seimbang.

#### **2. Pembangunan dan Evaluasi Model**
* **Model yang Dibandingkan:** Tiga algoritma klasifikasi digunakan dan dievaluasi:
    1.  `DecisionTreeClassifier` (sebagai baseline)
    2.  `RandomForestClassifier`
    3.  `GradientBoostingClassifier`
* **Evaluasi:** Kinerja model dievaluasi berdasarkan metrik **Akurasi, Presisi, Recall, dan F1-Score**, dengan F1-Score menjadi acuan utama karena kemampuannya menangani keseimbangan kelas.

#### **3. Optimisasi Model**
* Model dengan performa terbaik dari tahap sebelumnya (asumsi: Random Forest) dioptimalkan lebih lanjut menggunakan **`GridSearchCV`**.
* Proses ini secara sistematis mencari kombinasi hyperparameter terbaik (`n_estimators`, `max_depth`, dll.) untuk memaksimalkan `f1_score`.

---

## 🏁 Kesimpulan Akhir

Proyek ini berhasil menyelesaikan alur kerja machine learning end-to-end, dari data mentah hingga model prediktif yang siap digunakan.

1.  **Tahap Clustering** berhasil membuktikan bahwa meskipun pelanggan memiliki demografi (pelajar muda) dan perilaku transaksi (debit di cabang) yang serupa, mereka dapat disegmentasikan secara efektif berdasarkan **lokasi geografis**. Ini memberikan insight bisnis bahwa strategi pemasaran atau operasional cabang mungkin perlu disesuaikan per wilayah.

2.  **Tahap Klasifikasi** berhasil membangun dan mengoptimalkan model (Tuned Random Forest) yang mampu memprediksi segmen geografis pelanggan dengan **tingkat akurasi dan F1-score yang sangat tinggi**. Ini memberikan alat praktis bagi bisnis untuk secara otomatis mengkategorikan pelanggan baru dan menerapkan strategi yang relevan sejak dini.

Secara keseluruhan, proyek ini menunjukkan bagaimana kombinasi teknik *unsupervised* dan *supervised learning* dapat digunakan untuk menggali insight dari data dan mengubahnya menjadi sebuah model prediktif yang fungsional dan bernilai bisnis.

## 📁 Struktur File

-   `Clustering.ipynb`: Notebook untuk analisis eksplorasi, pra-pemrosesan, dan pembangunan model clustering.
-   `Klasifikasi.ipynb`: Notebook untuk pembangunan model klasifikasi berdasarkan hasil clustering.
-   `data_clustering.csv`: Output dari `Clustering.ipynb`. Berisi data yang sudah diproses (scaled & encoded) dan siap digunakan untuk model klasifikasi.
-   `data_clustering_inverse.csv`: Output dari `Clustering.ipynb`. Berisi data hasil clustering dengan nilai asli yang dapat dibaca manusia.
-   `*.h5`: Berbagai file model yang disimpan menggunakan `joblib`.
-   `README.md`: File ini.

## 🚀 Cara Menjalankan

1.  Pastikan semua library yang tercantum di `Import Library` pada kedua notebook telah terpasang di lingkungan Python Anda.
2.  Jalankan notebook `Clustering.ipynb` dari awal hingga akhir. Ini akan menghasilkan file `data_clustering.csv` yang dibutuhkan oleh notebook berikutnya.
3.  Jalankan notebook `Klasifikasi.ipynb` dari awal hingga akhir.