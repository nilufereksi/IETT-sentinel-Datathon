# ============================================================
# İETT SENTINEL — Streamlit Uygulaması
# Kurulum: pip install streamlit plotly pandas
# Çalıştır: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Sayfa Ayarları ────────────────────────────────────────
st.set_page_config(
    page_title="İETT SENTINEL",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tema / CSS ────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0d1117; }
  [data-testid="stSidebar"]          { background-color: #161b22; }
  .metric-card {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 10px; padding: 16px 20px;
      text-align: center;
  }
  .metric-val   { font-size: 28px; font-weight: 700; }
  .metric-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .red   { color: #e74c3c; }
  .green { color: #2ecc71; }
  .amber { color: #f39c12; }
  .blue  { color: #3498db; }
  .section-header {
      font-size: 18px; font-weight: 700; color: #f0f6fc;
      border-left: 4px solid #e74c3c;
      padding-left: 12px; margin: 24px 0 16px;
  }
</style>
""", unsafe_allow_html=True)

# ── Veri Yükleme ─────────────────────────────────────────
DATA_PATH = "data/"   # Streamlit Cloud için: CSV'leri repoya at

@st.cache_data
def yukle(dosya, **kwargs):
    yol = DATA_PATH + dosya
    if not os.path.exists(yol):
        return None
    return pd.read_csv(yol, **kwargs)

@st.cache_data
def tum_verileri_yukle():
    return {
        "rf_riskli"    : yukle("hucre12_bugun_riskli_araclar.csv"),
        "rf_skorlar"   : yukle("hucre12_rf_arac_skorlar.csv"),
        "rf_fi"        : yukle("hucre12_feature_importance.csv"),
        "arac_risk"    : yukle("modul1_arac_risk_kumeleri.csv"),
        "bakim"        : yukle("modul1_bakim_oncelik_listesi.csv"),
        "birliktelik"  : yukle("modul2_birliktelik_kurallari.csv"),
        "zayi"         : yukle("modul2_zayi_sefer_analizi.csv"),
        "denetim"      : yukle("modul2b_denetim_ozet.csv"),
        "kritik_hat"   : yukle("modul3_kritik_hatlar.csv"),
        "domino"       : yukle("modul3_domino_simulasyon.csv"),
        "hat_iptal"    : yukle("modul3_hat_iptal_profili.csv"),
        "ilce_risk"    : yukle("modul4_ilce_risk.csv"),
        "aylik_ariza"  : yukle("modul4_aylik_ariza_trend.csv"),
        "aylik_kaza"   : yukle("modul4_aylik_kaza_trend.csv"),
        "gunluk"       : yukle("modul5_gunluk_anomali.csv"),
        "etkinlik"     : yukle("modul5_etkinlik_adaylari.csv"),
    }

veri = tum_verileri_yukle()

RENK_RF = {
    "🔴 Kritik" : "#e74c3c",
    "🟠 Yüksek" : "#e67e22",
    "🟡 Orta"   : "#f1c40f",
    "🟢 Düşük"  : "#2ecc71",
}
DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#f0f6fc"),
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚨 İETT SENTINEL")
    st.markdown("*Akıllı Kriz Öngörü Sistemi*")
    st.divider()

    sayfa = st.radio(
        "Navigasyon",
        options=[
            "🏠 Ana Sayfa",
            "🔴 Risk Analizi",
            "🕸️ Domino Etkisi",
            "🗺️ Kriz Haritası",
            "📈 Trend Analizi",
            "⚠️ Erken Uyarı",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#8b949e;'>
    <b>Model:</b> Random Forest<br>
    <b>AUC:</b> 0.885 &nbsp;|&nbsp; <b>CV:</b> 0.879<br>
    <b>Veri:</b> 2024-2025<br>
    <b>Araç:</b> 6,769 &nbsp;|&nbsp; <b>Kural:</b> 380
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🏠 ANA SAYFA
# ════════════════════════════════════════════════════════════
if sayfa == "🏠 Ana Sayfa":
    st.markdown(
        "<h1 style='color:#e74c3c; margin-bottom:4px;'>🚨 İETT SENTINEL</h1>"
        "<p style='color:#8b949e; font-size:16px;'>Arıza olmadan önce tahmin et — "
        "kriz yayılmadan önce önle</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # KPI Kartları
    rf  = veri["rf_riskli"]
    ars = veri["arac_risk"]
    dom = veri["domino"]

    c1, c2, c3, c4 = st.columns(4)
    kritik_n  = int((rf["RF_ARIZA_SKORU"] >= 75).sum()) if rf is not None else 0
    zayi_top  = int(ars["toplam_zayi_sefer"].sum()) if ars is not None else 0
    domino_mx = int(dom["etkilenen_hat"].max()) if dom is not None else 0

    c1.markdown(f"""<div class="metric-card">
        <div class="metric-val red">{kritik_n:,}</div>
        <div class="metric-label">🔴 Kritik Araç (RF ≥75%)</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""<div class="metric-card">
        <div class="metric-val amber">{zayi_top:,}</div>
        <div class="metric-label">🚫 Toplam Zayi Sefer</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""<div class="metric-card">
        <div class="metric-val blue">{domino_mx}</div>
        <div class="metric-label">🕸️ Maks. Domino Etkisi (hat)</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""<div class="metric-card">
        <div class="metric-val green">0.885</div>
        <div class="metric-label">🤖 RF Model AUC</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>SENTINEL'in 3 Temel Çıktısı</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-color:#e74c3c;">
        <div style="font-size:32px;">🔧</div>
        <div style="font-size:15px; font-weight:700; color:#e74c3c; margin:8px 0;">
            Bakım Öncelik Listesi</div>
        <div style="font-size:13px; color:#c9d1d9;">
            RF modeli ile <b>2,875 araç</b> kritik seviyede.<br>
            Hangi aracın yarın arıza yapacağını bugünden biliriz.</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-color:#f39c12;">
        <div style="font-size:32px;">⚠️</div>
        <div style="font-size:15px; font-weight:700; color:#f39c12; margin:8px 0;">
            Erken Uyarı Sistemi</div>
        <div style="font-size:13px; color:#c9d1d9;">
            Olumsuz denetimden sonra <b>14 gün içinde %71.3</b> arıza.<br>
            Denetim verisi → otomatik bakım bildirimi.</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-color:#3498db;">
        <div style="font-size:32px;">🕸️</div>
        <div style="font-size:15px; font-weight:700; color:#3498db; margin:8px 0;">
            Domino Kriz Yönetimi</div>
        <div style="font-size:13px; color:#c9d1d9;">
            Kritik hat kapanırsa <b>183 hat</b> etkilenir.<br>
            Hat bağımlılık ağı ile kriz öncesi müdahale.</div>
        </div>""", unsafe_allow_html=True)

    # Mini RF Risk Pie
    st.markdown("<div class='section-header'>Anlık Risk Durumu</div>",
                unsafe_allow_html=True)

    if veri["rf_skorlar"] is not None:
        sk = veri["rf_skorlar"]
        sk["RF_RISK_ETIKETI"] = sk["RF_RISK_ETIKETI"].astype(str)
        dag = sk["RF_RISK_ETIKETI"].value_counts().reset_index()
        dag.columns = ["Risk", "Adet"]

        fig = px.pie(dag, names="Risk", values="Adet",
                     color="Risk", color_discrete_map=RENK_RF,
                     hole=0.5, title="RF Arıza Risk Dağılımı — 6,769 Araç")
        fig.update_layout(**DARK, height=350)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 🔴 RİSK ANALİZİ
# ════════════════════════════════════════════════════════════
elif sayfa == "🔴 Risk Analizi":
    st.markdown("<h2 style='color:#e74c3c;'>🔴 Araç Risk Analizi</h2>",
                unsafe_allow_html=True)

    rf = veri["rf_riskli"]
    sk = veri["rf_skorlar"]

    if rf is None:
        st.error("Veri bulunamadı.")
        st.stop()

    # Filtreler
    col1, col2, col3 = st.columns(3)
    with col1:
        esik = st.slider("RF Risk Eşiği (%)", 50, 100, 75, 5)
    with col2:
        markalar = ["Tümü"] + sorted(rf["marka"].dropna().unique().tolist())
        secili_marka = st.selectbox("Marka", markalar)
    with col3:
        ilceler = ["Tümü"] + sorted(rf["garaj_ilce"].dropna().unique().tolist())
        secili_ilce = st.selectbox("Garaj İlçesi", ilceler)

    # Filtrele
    filtre = rf["RF_ARIZA_SKORU"] >= esik
    if secili_marka != "Tümü":
        filtre &= rf["marka"] == secili_marka
    if secili_ilce != "Tümü":
        filtre &= rf["garaj_ilce"] == secili_ilce

    df_filtre = rf[filtre].sort_values("RF_ARIZA_SKORU", ascending=False)

    # KPI satırı
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Riskli Araç", f"{len(df_filtre):,}")
    c2.metric("Ort. RF Skoru", f"%{df_filtre['RF_ARIZA_SKORU'].mean():.1f}")
    c3.metric("Toplam Zayi Sefer", f"{int(df_filtre['toplam_zayi_sefer'].sum()):,}")
    c4.metric("Toplam Çekici Talebi", f"{int(df_filtre['cekici_talep_sayisi'].sum()):,}")

    st.divider()

    # Grafik + Tablo yan yana
    gcol1, gcol2 = st.columns([1, 1])

    with gcol1:
        st.markdown("#### RF Skoru Dağılımı (Top 20)")
        top20 = df_filtre.head(20)
        fig = px.bar(
            top20, x="RF_ARIZA_SKORU",
            y=top20["SIDARAC"].astype(str),
            orientation="h",
            color="RF_ARIZA_SKORU",
            color_continuous_scale="RdYlGn_r",
            range_color=[50, 100],
            labels={"RF_ARIZA_SKORU": "Risk Skoru (%)",
                    "y": "Araç ID"},
        )
        fig.update_layout(**DARK, height=500, showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with gcol2:
        st.markdown("#### Garaj İlçesi × Kritik Araç")
        garaj_g = (df_filtre.groupby("garaj_ilce")
                   .size().reset_index(name="sayi")
                   .sort_values("sayi", ascending=False))
        fig2 = px.bar(garaj_g, x="garaj_ilce", y="sayi",
                      color="sayi", color_continuous_scale="Reds",
                      labels={"garaj_ilce": "Garaj İlçesi",
                              "sayi": "Kritik Araç"})
        fig2.update_layout(**DARK, height=500,
                           coloraxis_showscale=False)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    # Detay tablosu
    st.markdown("#### Riskli Araç Listesi")
    goster_cols = ["SIDARAC", "marka", "arac_yasi", "yas_kategori",
                   "toplam_ariza", "toplam_zayi_sefer", "cekici_talep_sayisi",
                   "olumsuz_denetim", "RF_ARIZA_SKORU", "RF_RISK_ETIKETI",
                   "garaj", "garaj_ilce"]
    goster_cols = [c for c in goster_cols if c in df_filtre.columns]
    st.dataframe(
        df_filtre[goster_cols].reset_index(drop=True),
        use_container_width=True, height=350,
    )
    st.download_button(
        "⬇️ CSV İndir",
        df_filtre[goster_cols].to_csv(index=False).encode("utf-8"),
        "riskli_araclar.csv", "text/csv",
    )

    # Feature Importance
    st.divider()
    st.markdown("#### RF — Feature Importance")
    fi = veri["rf_fi"]
    if fi is not None:
        fig3 = px.bar(
            fi.head(12).sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="RdYlGn",
            labels={"importance": "Önem", "feature": "Özellik"},
        )
        fig3.update_layout(**DARK, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 🕸️ DOMİNO ETKİSİ
# ════════════════════════════════════════════════════════════
elif sayfa == "🕸️ Domino Etkisi":
    st.markdown("<h2 style='color:#3498db;'>🕸️ Hat Bağımlılık & Domino Etkisi</h2>",
                unsafe_allow_html=True)

    dom  = veri["domino"]
    khat = veri["kritik_hat"]
    hi   = veri["hat_iptal"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Domino Simülasyonu — Hat Kapanırsa")
        if dom is not None:
            fig = px.bar(
                dom.sort_values("etkilenen_hat", ascending=False),
                x="hat_adi", y="etkilenen_hat",
                color="etkilenen_hat",
                color_continuous_scale="Reds",
                text="etkilenen_hat",
                labels={"hat_adi": "Hat",
                        "etkilenen_hat": "Etkilenen Hat Sayısı"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(**DARK, height=420,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Domino verisi bulunamadı.")

    with col2:
        st.markdown("#### En Kritik Hatlar (Betweenness)")
        if khat is not None:
            fig2 = px.bar(
                khat.sort_values("betweenness", ascending=True).tail(15),
                x="betweenness", y="hat_adi", orientation="h",
                color="derece",
                color_continuous_scale="RdYlGn_r",
                labels={"betweenness": "Betweenness",
                        "hat_adi": "Hat",
                        "derece": "Bağlantı"},
            )
            fig2.update_layout(**DARK, height=420,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Kritik hat verisi bulunamadı.")

    # Hat iptal oranı
    st.divider()
    st.markdown("#### Hat Bazında İptal Oranı (Top 20)")
    if hi is not None:
        hi_filtre = (hi[hi["toplam_sefer"] >= 100]
                     .sort_values("IPTAL_ORANI", ascending=False)
                     .head(20))
        fig3 = px.bar(
            hi_filtre, x="HATKODU", y="IPTAL_ORANI",
            color="IPTAL_ORANI", color_continuous_scale="Reds",
            labels={"HATKODU": "Hat Kodu",
                    "IPTAL_ORANI": "İptal Oranı"},
        )
        fig3.update_layout(**DARK, height=380,
                           coloraxis_showscale=False)
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# 🗺️ KRİZ HARİTASI
# ════════════════════════════════════════════════════════════
elif sayfa == "🗺️ Kriz Haritası":
    st.markdown("<h2 style='color:#e67e22;'>🗺️ İstanbul Kriz Haritası</h2>",
                unsafe_allow_html=True)

    ilce = veri["ilce_risk"]

    if ilce is not None:
        ilce_f = ilce[ilce["ILCE"] != "BELİRSİZ"].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### İlçe Bazında Kaza Sayısı")
            fig1 = px.bar(
                ilce_f.sort_values("kaza_sayisi", ascending=False).head(15),
                x="ILCE", y="kaza_sayisi",
                color="kriz_orani",
                color_continuous_scale="RdYlGn_r",
                labels={"ILCE": "İlçe",
                        "kaza_sayisi": "Kaza Sayısı",
                        "kriz_orani": "Kriz Oranı"},
            )
            fig1.update_layout(**DARK, height=420,
                               coloraxis_showscale=True)
            fig1.update_xaxes(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("#### Kriz Oranı — Bubble Chart")
            fig2 = px.scatter(
                ilce_f.head(20),
                x="kaza_sayisi", y="kriz_orani",
                size="toplam_yarali",
                color="kriz_orani",
                color_continuous_scale="RdYlGn_r",
                hover_name="ILCE",
                text="ILCE",
                labels={"kaza_sayisi": "Kaza Sayısı",
                        "kriz_orani": "Kriz Oranı",
                        "toplam_yarali": "Yaralı"},
            )
            fig2.update_traces(textposition="top center",
                               textfont_size=9)
            fig2.update_layout(**DARK, height=420,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Detay tablo
        st.markdown("#### İlçe Risk Detayı")
        st.dataframe(
            ilce_f.sort_values("kaza_sayisi", ascending=False)
                  .reset_index(drop=True),
            use_container_width=True, height=300,
        )
    else:
        st.info("İlçe risk verisi bulunamadı.")

# ════════════════════════════════════════════════════════════
# 📈 TREND ANALİZİ
# ════════════════════════════════════════════════════════════
elif sayfa == "📈 Trend Analizi":
    st.markdown("<h2 style='color:#2ecc71;'>📈 Arıza & Kaza Zaman Serisi</h2>",
                unsafe_allow_html=True)

    aa = veri["aylik_ariza"]
    ak = veri["aylik_kaza"]
    g  = veri["gunluk"]

    if aa is not None:
        aa["TARIH"] = pd.to_datetime(aa["TARIH"])
        fig1 = go.Figure()
        for yil, renk in [(2024, "#3498db"), (2025, "#e74c3c")]:
            d = aa[aa["YIL"] == yil]
            fig1.add_trace(go.Scatter(
                x=d["TARIH"], y=d["ariza_sayisi"],
                name=f"Arıza {yil}",
                line=dict(color=renk, width=2),
                mode="lines+markers", marker=dict(size=5),
            ))
        fig1.update_layout(**DARK, title="Aylık Arıza Trendi (2024 vs 2025)",
                           height=380)
        st.plotly_chart(fig1, use_container_width=True)

    if ak is not None:
        ak["TARIH"] = pd.to_datetime(ak["TARIH"])
        fig2 = go.Figure()
        for yil, renk in [(2024, "#2ecc71"), (2025, "#f39c12")]:
            d = ak[ak["KAZA_YIL"] == yil]
            fig2.add_trace(go.Scatter(
                x=d["TARIH"], y=d["kaza_sayisi"],
                name=f"Kaza {yil}",
                line=dict(color=renk, width=2),
                mode="lines+markers", marker=dict(size=5),
            ))
        fig2.update_layout(**DARK, title="Aylık Kaza Trendi (2024 vs 2025)",
                           height=380)
        st.plotly_chart(fig2, use_container_width=True)

    if g is not None:
        g["TARIH"] = pd.to_datetime(g["TARIH"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=g["TARIH"], y=g["sefer_sayisi"],
            mode="lines", name="Günlük Sefer",
            line=dict(color="#3498db", width=1),
        ))
        an = g[g["SEFER_ANOMALI"] == True]
        fig3.add_trace(go.Scatter(
            x=an["TARIH"], y=an["sefer_sayisi"],
            mode="markers", name="Anomali",
            marker=dict(color="#e74c3c", size=10, symbol="star"),
        ))
        fig3.update_layout(**DARK,
                           title="Günlük Sefer Sayısı & Anomaliler",
                           height=380)
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════
# ⚠️ ERKEN UYARI
# ════════════════════════════════════════════════════════════
elif sayfa == "⚠️ Erken Uyarı":
    st.markdown("<h2 style='color:#f39c12;'>⚠️ Denetim → Arıza Erken Uyarı</h2>",
                unsafe_allow_html=True)

    den = veri["denetim"]
    zay = veri["zayi"]
    bir = veri["birliktelik"]

    # Metrik satırı
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Denetim-Arıza Korelasyon", "r = 0.545")
    c2.metric("14 Gün İçi Arıza Oranı",   "%71.3")
    c3.metric("Olumsuz Denetim Oranı (Ort)", "%38.2")
    c4.metric("Birliktelik Kural Sayısı",  "380")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Olumsuz Denetim Oranı Dağılımı")
        if den is not None:
            den_f = den[den["toplam_denetim"] >= 5]
            fig1 = px.histogram(
                den_f, x="OLUMSUZ_ORAN", nbins=40,
                color_discrete_sequence=["#e67e22"],
                labels={"OLUMSUZ_ORAN": "Olumsuz Oran"},
            )
            fig1.update_layout(**DARK, height=380)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Sefer İptaline En Çok Yol Açan Arızalar")
        if zay is not None:
            zay_r = zay.reset_index() if "ARIZAUSTKODTANIM" not in zay.columns \
                    else zay
            fig2 = px.bar(
                zay_r.head(10).sort_values("toplam_zayi_sefer"),
                x="toplam_zayi_sefer", y="ARIZAUSTKODTANIM",
                orientation="h",
                color="ort_zayi_sefer",
                color_continuous_scale="Reds",
                labels={"ARIZAUSTKODTANIM": "Arıza Türü",
                        "toplam_zayi_sefer": "Zayi Sefer",
                        "ort_zayi_sefer": "Ort. Zayi"},
            )
            fig2.update_layout(**DARK, height=380,
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Birliktelik kuralları tablosu
    st.divider()
    st.markdown("#### Arıza Birliktelik Kuralları (Lift > 1.2)")
    if bir is not None:
        bir_f = (bir[bir["lift"] > 1.2]
                 .sort_values("lift", ascending=False)
                 .head(20)
                 .reset_index(drop=True))
        st.dataframe(bir_f, use_container_width=True, height=350)
    else:
        st.info("Birliktelik verisi bulunamadı.")