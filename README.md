# 🚨 İETT SENTINEL — Akıllı Kriz Öngörü Sistemi 

![20260413-1003-24 6423007-ezgif com-speed](https://github.com/user-attachments/assets/4df31e55-ef5e-47d2-b7f9-3f8b3509c1b3)

> İstanbul toplu ulaşımında arıza olmadan önce tahmin et — kriz yayılmadan önce önle.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Tech Istanbul-IETT Datathon Projesi Hakkında

**İETT SENTINEL**, İstanbul Elektrik Tramvay ve Tünel (İETT) filosundaki **6.769 aracı** ve **870 hattı** kapsayan, makine öğrenmesi tabanlı bir **proaktif kriz yönetim sistemidir**.

Random Forest modeli (AUC = 0.885) kullanarak araç arızalarını 30 gün önceden tahmin eder, hat aksama senaryolarını simüle eder ve erken uyarı sinyalleri üretir.

---

## ✨ Özellikler

| Modül | Açıklama |
|---|---|
| 🏠 **Ana Sayfa** | Filo geneli KPI'lar ve anlık risk dağılımı |
| 🔴 **Risk Analizi** | RF modeli ile araç bazlı 30 günlük arıza olasılığı |
| 🕸️ **Domino Etkisi** | Hat aksama simülatörü + İstanbul harita görselleştirmesi |
| 🗺️ **Kriz Haritası** | İlçe bazlı kaza yoğunluğu ve kriz tetikleyici analizi |
| 📈 **Trend Analizi** | 2024–2025 arıza & kaza karşılaştırması + anomali tespiti |
| ⚠️ **Erken Uyarı** | Denetim verisi → arıza örüntüsü (14 günlük pencere) |

---

## 📊 Model Performansı

| Metrik | Değer |
|---|---|
| ROC-AUC | **0.885** |
| Cross-Validation AUC | **0.879** |
| Veri Dönemi | 2024 – 2025 |
| Toplam Araç | 6,769 |
| Toplam Hat | 870 |
| Arıza Birliktelik Kuralı | 380 |

**En Güçlü Özellikler:**
- Toplam arıza sayısı
- Benzersiz arıza türü sayısı
- Olumsuz denetim oranı
- Çekici talep sayısı
- Araç yaşı

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.9+
- pip

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullaniciadi/sentinel.git
cd sentinel

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. Veri dosyalarını hazırla
# data.zip dosyasını aç — data/ klasörü otomatik oluşacak
# (veya zip'i manuel olarak data/ klasörüne çıkar)

# 4. Uygulamayı başlat
streamlit run app.py
```

Uygulama varsayılan olarak **http://localhost:8501** adresinde çalışır.

---

## 📁 Proje Yapısı

```
sentinel/
├── app.py                  # Ana Streamlit uygulaması
├── requirements.txt        # Python bağımlılıkları
├── data.zip                # Ham veri arşivi
└── data/                   # CSV veri dosyaları (zip'ten çıkarılır)
    ├── hucre12_bugun_riskli_araclar.csv
    ├── hucre12_rf_arac_skorlar.csv
    ├── hucre12_feature_importance.csv
    ├── hucre13_alternatif_hat_onerileri.csv
    ├── modul1_arac_risk_kumeleri.csv
    ├── modul1_bakim_oncelik_listesi.csv
    ├── modul2_birliktelik_kurallari.csv
    ├── modul2_zayi_sefer_analizi.csv
    ├── modul2b_denetim_ozet.csv
    ├── modul3_kritik_hatlar.csv
    ├── modul3_domino_simulasyon.csv
    ├── modul3_hat_iptal_profili.csv
    ├── modul4_ilce_risk.csv
    ├── modul4_aylik_ariza_trend.csv
    ├── modul4_aylik_kaza_trend.csv
    ├── modul5_gunluk_anomali.csv
    └── hat_guzergah_koordinatlar.csv
```

---

## 🧰 Teknoloji Yığını

- **[Streamlit](https://streamlit.io/)** — Web arayüzü
- **[Pandas](https://pandas.pydata.org/)** — Veri manipülasyonu
- **[Plotly](https://plotly.com/)** — İnteraktif görselleştirme
- **[Scikit-learn](https://scikit-learn.org/)** — Random Forest modeli
- **[NetworkX](https://networkx.org/)** — Hat bağımlılık ağı analizi

---

## 🔍 Temel Bulgular

- 📌 Olumsuz denetim sonrası araçların **%71.3'ü 14 gün içinde** arıza yapmaktadır
- 📌 Denetim olumsuzluk oranı ile arıza sayısı arasında **r = 0.545** korelasyon
- 📌 **AVR1** hattı kapanırsa bağımlılık ağındaki maksimum **183 hat** etkilenmektedir
- 📌 Bugün itibarıyla **2.875 araç** kritik risk eşiğinin (RF ≥ %75) üzerindedir

---

## 📄 Lisans

Bu proje [GPL-3.0](LICENSE) ile lisanslanmıştır.
