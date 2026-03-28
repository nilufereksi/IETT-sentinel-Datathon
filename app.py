# ============================================================
# İETT SENTINEL — Streamlit Uygulaması (Final)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="İETT SENTINEL",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8F9FA; }
[data-testid="stSidebar"]          { background: #FFFFFF; border-right: 1px solid #E9ECEF; }
[data-testid="stHeader"]           { background: transparent; }

.kpi-card {
    background: #FFFFFF; border: 1px solid #E9ECEF;
    border-radius: 12px; padding: 20px 24px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-val   { font-size: 30px; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: 12px; color: #6C757D; margin-top: 6px; font-weight: 500; }

.info-box {
    background: #EFF6FF; border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin-bottom: 20px; font-size: 14px;
    color: #1E40AF; line-height: 1.6;
}
.warn-box {
    background: #FFFBEB; border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin-bottom: 20px; font-size: 14px;
    color: #92400E; line-height: 1.6;
}
.success-box {
    background: #F0FDF4; border-left: 4px solid #22C55E;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    margin-bottom: 20px; font-size: 14px;
    color: #166534; line-height: 1.6;
}
.section-title {
    font-size: 15px; font-weight: 700; color: #1A1A2E;
    margin: 24px 0 12px; padding-bottom: 8px;
    border-bottom: 2px solid #E9ECEF;
}
.output-card {
    background: #FFFFFF; border: 1px solid #E9ECEF;
    border-radius: 12px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.oneri-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px; border-left: 5px solid #2563EB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.model-badge {
    background: #F1F5F9; border-radius: 8px;
    padding: 10px 12px; font-size: 12px;
    color: #475569; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FAFAFA",
    font=dict(color="#1A1A2E", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)

DATA = "data/"

@st.cache_data
def yukle(f, **kw):
    p = DATA + f
    return pd.read_csv(p, **kw) if os.path.exists(p) else None

@st.cache_data
def tum_veri():
    return {
        "rf_riskli"   : yukle("hucre12_bugun_riskli_araclar.csv"),
        "rf_skorlar"  : yukle("hucre12_rf_arac_skorlar.csv"),
        "rf_fi"       : yukle("hucre12_feature_importance.csv"),
        "arac_risk"   : yukle("modul1_arac_risk_kumeleri.csv"),
        "bakim"       : yukle("modul1_bakim_oncelik_listesi.csv"),
        "birliktelik" : yukle("modul2_birliktelik_kurallari.csv"),
        "zayi"        : yukle("modul2_zayi_sefer_analizi.csv"),
        "denetim"     : yukle("modul2b_denetim_ozet.csv"),
        "kritik_hat"  : yukle("modul3_kritik_hatlar.csv"),
        "domino"      : yukle("modul3_domino_simulasyon.csv"),
        "hat_iptal"   : yukle("modul3_hat_iptal_profili.csv"),
        "ilce_risk"   : yukle("modul4_ilce_risk.csv"),
        "aylik_ariza" : yukle("modul4_aylik_ariza_trend.csv"),
        "aylik_kaza"  : yukle("modul4_aylik_kaza_trend.csv"),
        "gunluk"      : yukle("modul5_gunluk_anomali.csv"),
        "oneri"       : yukle("hucre13_alternatif_hat_onerileri.csv"),
    }

V = tum_veri()

RENK = {
    "🔴 Kritik": "#DC2626", "🟠 Yüksek": "#EA580C",
    "🟡 Orta"  : "#CA8A04", "🟢 Düşük" : "#16A34A",
}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚨 İETT SENTINEL")
    st.caption("Akıllı Kriz Öngörü Sistemi")
    st.divider()
    sayfa = st.radio("", [
        "🏠  Ana Sayfa",
        "🔴  Risk Analizi",
        "🕸️  Domino Etkisi",
        "🗺️  Kriz Haritası",
        "📈  Trend Analizi",
        "⚠️  Erken Uyarı",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div class="model-badge">
    <b>Model:</b> Random Forest<br>
    <b>ROC-AUC:</b> 0.885 &nbsp;|&nbsp; <b>CV:</b> 0.879<br>
    <b>Dönem:</b> 2024–2025<br>
    <b>Araç:</b> 6,769 &nbsp;|&nbsp; <b>Hat:</b> 870
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 🏠 ANA SAYFA
# ════════════════════════════════════════════════════════════
if "Ana Sayfa" in sayfa:
    st.markdown("## 🚨 İETT SENTINEL")
    st.markdown("**İstanbul toplu ulaşımında arıza olmadan önce tahmin et — kriz yayılmadan önce önle.**")
    st.divider()

    rf  = V["rf_riskli"]
    ar  = V["arac_risk"]
    dom = V["domino"]
    sk  = V["rf_skorlar"]

    # Dinamik KPI — CSV'den hesapla
    kritik = int((rf["RF_ARIZA_SKORU"] >= 75).sum()) if rf is not None else 0
    zayi   = int(ar["toplam_zayi_sefer"].sum())       if ar is not None else 0
    domino = int(dom["etkilenen_hat"].max())           if dom is not None else 0

    # ── KPI Satırı ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#DC2626">{kritik:,}</div><div class="kpi-label">Kritik Araç (RF ≥75%)</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#EA580C">{zayi:,}</div><div class="kpi-label">Zayi Sefer (2024–25)</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#2563EB">{domino:,}</div><div class="kpi-label">Maks. Domino Etkisi (hat)</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#16A34A">0.885</div><div class="kpi-label">RF Model AUC</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 Temel Çıktı — Dinamik ──────────────────────────
    st.markdown('<div class="section-title">3 Temel Çıktı</div>', unsafe_allow_html=True)

    # Dinamik değerler
    en_kritik_hat = dom.iloc[dom["etkilenen_hat"].idxmax()]["hat_adi"] if dom is not None else "—"

    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(f"""
        <div class="output-card" style="border-top:4px solid #DC2626;">
          <div style="font-size:28px;margin-bottom:8px">🔧</div>
          <div style="font-size:15px;font-weight:700;color:#DC2626;margin-bottom:8px">
            Bakım Öncelik Listesi</div>
          <div style="font-size:13px;color:#495057;line-height:1.7">
            RF modeli <b style="color:#DC2626">{kritik:,} aracı</b> kritik olarak tespit etti.<br>
            Hangi araç yarın arıza yapacak? Bugünden biliyoruz.
          </div>
        </div>""", unsafe_allow_html=True)
    with o2:
        st.markdown("""
        <div class="output-card" style="border-top:4px solid #CA8A04;">
          <div style="font-size:28px;margin-bottom:8px">⚠️</div>
          <div style="font-size:15px;font-weight:700;color:#CA8A04;margin-bottom:8px">
            14 Gün Erken Uyarı</div>
          <div style="font-size:13px;color:#495057;line-height:1.7">
            Olumsuz denetimden sonra araçların
            <b style="color:#CA8A04">%71.3'ü 14 gün içinde</b>
            arıza yaptı.<br>Denetim verisi → otomatik bakım bildirimi.
          </div>
        </div>""", unsafe_allow_html=True)
    with o3:
        st.markdown(f"""
        <div class="output-card" style="border-top:4px solid #2563EB;">
          <div style="font-size:28px;margin-bottom:8px">🕸️</div>
          <div style="font-size:15px;font-weight:700;color:#2563EB;margin-bottom:8px">
            Domino Kriz Yönetimi</div>
          <div style="font-size:13px;color:#495057;line-height:1.7">
            <b style="color:#2563EB">{en_kritik_hat}</b> kapanırsa
            <b style="color:#2563EB">{domino:,} hat</b> etkilenir.<br>
            Alternatif hat önerisi ile kriz yayılmadan müdahale.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Dağılımı — Emoji Temizlenmiş ────────────────
    st.markdown('<div class="section-title">Filodaki Anlık Risk Dağılımı</div>', unsafe_allow_html=True)

    if sk is not None:
        sk = sk.copy()
        sk["RF_RISK_ETIKETI"] = sk["RF_RISK_ETIKETI"].astype(str)

        # Emoji temizle — pie label için
        RENK_TEMIZ = {
            "Kritik" : "#DC2626",
            "Yüksek" : "#EA580C",
            "Orta"   : "#CA8A04",
            "Düşük"  : "#16A34A",
        }
        def etiket_temizle(s):
            for emoji in ["🔴 ","🟠 ","🟡 ","🟢 "]:
                s = s.replace(emoji, "")
            return s.strip()

        sk["ETIKET_TEMIZ"] = sk["RF_RISK_ETIKETI"].apply(etiket_temizle)
        dag = sk["ETIKET_TEMIZ"].value_counts().reset_index()
        dag.columns = ["Risk", "Adet"]

        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            fig = px.pie(dag, names="Risk", values="Adet",
                         color="Risk",
                         color_discrete_map=RENK_TEMIZ,
                         hole=0.55)
            fig.update_traces(
                textinfo="label+percent",
                textfont_size=13,
                textposition="outside",
            )
            fig.update_layout(
                **LIGHT, height=340,
                showlegend=False,
                title="6,769 araç — RF Arıza Risk Dağılımı",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_tbl:
            st.markdown("<br>", unsafe_allow_html=True)
            RENK_SOL = {
                "Kritik": "#DC2626", "Yüksek": "#EA580C",
                "Orta"  : "#CA8A04", "Düşük" : "#16A34A",
            }
            toplam = dag["Adet"].sum()
            for _, r in dag.iterrows():
                pct  = r["Adet"] / toplam * 100
                renk = RENK_SOL.get(r["Risk"], "#aaa")
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                     align-items:center;padding:12px 16px;margin-bottom:8px;
                     background:#F8F9FA;border-radius:8px;
                     border-left:4px solid {renk};">
                  <span style="font-weight:600;color:#1A1A2E">{r["Risk"]}</span>
                  <span style="color:#6C757D;font-size:13px">
                    {r["Adet"]:,} araç &nbsp;·&nbsp; %{pct:.1f}
                  </span>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 🔴 RİSK ANALİZİ
# ════════════════════════════════════════════════════════════
elif "Risk Analizi" in sayfa:
    st.markdown("## 🔴 Araç Risk Analizi")
    st.markdown('<div class="info-box">🤖 <b>Random Forest modeli</b> (AUC=0.885) — 17 özellik kullanılarak her araç için <b>30 günlük arıza olasılığı</b> hesaplandı. Filtreler ile garaj, marka ve risk eşiğine göre sorgulayabilirsiniz.</div>', unsafe_allow_html=True)

    rf = V["rf_riskli"]
    if rf is None:
        st.error("Veri bulunamadı.")
        st.stop()

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        esik = st.slider("Risk Eşiği (%)", 50, 100, 75, 5)
    with fc2:
        markalar = ["Tümü"] + sorted(rf["marka"].dropna().unique().tolist())
        marka = st.selectbox("Marka", markalar)
    with fc3:
        ilceler = ["Tümü"] + sorted(rf["garaj_ilce"].dropna().unique().tolist())
        ilce = st.selectbox("Garaj İlçesi", ilceler)

    f = rf["RF_ARIZA_SKORU"] >= esik
    if marka != "Tümü": f &= rf["marka"] == marka
    if ilce  != "Tümü": f &= rf["garaj_ilce"] == ilce
    df = rf[f].sort_values("RF_ARIZA_SKORU", ascending=False)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Riskli Araç",      f"{len(df):,}")
    m2.metric("Ort. Risk Skoru",  f"%{df['RF_ARIZA_SKORU'].mean():.1f}" if len(df) > 0 else "—")
    m3.metric("Toplam Zayi Sefer",f"{int(df['toplam_zayi_sefer'].sum()):,}" if len(df) > 0 else "—")
    m4.metric("Çekici Talebi",    f"{int(df['cekici_talep_sayisi'].sum()):,}" if len(df) > 0 else "—")
    st.divider()

    st.markdown('<div class="section-title">En Riskli 20 Araç</div>', unsafe_allow_html=True)
    if len(df) > 0:
        top = df.head(20).copy()
        top["Araç"] = (top["SIDARAC"].astype(int).astype(str)
                       + "  ·  " + top["marka"].fillna("")
                       + "  ·  " + top["garaj_ilce"].fillna(""))
        fig = px.bar(top, x="RF_ARIZA_SKORU", y="Araç",
                     orientation="h",
                     color="RF_ARIZA_SKORU",
                     color_continuous_scale=["#16A34A","#CA8A04","#DC2626"],
                     range_color=[50, 100],
                     labels={"RF_ARIZA_SKORU": "Risk Skoru (%)"},
                     text="RF_ARIZA_SKORU")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(**LIGHT, height=580,
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Seçilen kriterlere uyan araç bulunamadı.")

    st.markdown('<div class="section-title">Garaj İlçesi Bazında Dağılım</div>', unsafe_allow_html=True)
    if len(df) > 0:
        garaj = (df.groupby("garaj_ilce").size()
                   .reset_index(name="Araç Sayısı")
                   .sort_values("Araç Sayısı", ascending=False))
        fig2 = px.bar(garaj, x="garaj_ilce", y="Araç Sayısı",
                      color="Araç Sayısı",
                      color_continuous_scale=["#FEE2E2","#DC2626"],
                      labels={"garaj_ilce":"Garaj İlçesi"},
                      text="Araç Sayısı")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**LIGHT, height=380, coloraxis_showscale=False)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Araç Listesi</div>', unsafe_allow_html=True)
    cols = ["SIDARAC","marka","arac_yasi","yas_kategori",
            "toplam_ariza","toplam_zayi_sefer","cekici_talep_sayisi",
            "olumsuz_denetim","RF_ARIZA_SKORU","RF_RISK_ETIKETI","garaj_ilce"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols].reset_index(drop=True),
                 use_container_width=True, height=320)
    if len(df) > 0:
        st.download_button("⬇️  CSV İndir",
                           df[cols].to_csv(index=False).encode("utf-8"),
                           "riskli_araclar.csv", "text/csv")

    fi = V["rf_fi"]
    if fi is not None:
        st.markdown('<div class="section-title">Modeli Etkileyen Özellikler</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">📊 Modelin kararında en çok etkili olan özellikler. <b>Toplam arıza sayısı</b> ve <b>benzersiz arıza türü</b> en güçlü göstergeler.</div>', unsafe_allow_html=True)
        fig3 = px.bar(fi.head(12).sort_values("importance"),
                      x="importance", y="feature", orientation="h",
                      color="importance",
                      color_continuous_scale=["#BFDBFE","#1D4ED8"],
                      labels={"importance":"Önem Skoru","feature":"Özellik"},
                      text="importance")
        fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig3.update_layout(**LIGHT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# 🕸️ DOMİNO ETKİSİ — Harita Entegreli (Final)
# Bu bloğu app.py'deki mevcut "elif "Domino" in sayfa:" bloğuyla değiştir
# ════════════════════════════════════════════════════════════
elif "Domino" in sayfa:
    st.markdown("## 🕸️ Hat Aksama Simülatörü")
    st.markdown('<div class="info-box">🔗 <b>794 hatlık GTFS bazlı bağımlılık ağı</b> ile bir hat aksar ya da kapanırsa hangi hatların etkileneceği ve <b>hangi alternatif hatların devreye alınabileceği</b> gerçek durak verisiyle hesaplandı.</div>', unsafe_allow_html=True)

    oneri = V["oneri"]
    dom   = V["domino"]
    khat  = V["kritik_hat"]
    hi    = V["hat_iptal"]

    # Güzergah koordinatları yükle
    @st.cache_data
    def yukle_guzergah():
        import ast
        df = yukle("hat_guzergah_koordinatlar.csv")
        if df is not None:
            df['koordinatlar'] = df['koordinatlar'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        return df

    guzergah_df = yukle_guzergah()

    # ── İnteraktif simülatör ─────────────────────────────
    st.markdown('<div class="section-title">Hat Aksama Simülatörü</div>', unsafe_allow_html=True)

    if oneri is not None and hi is not None:
        hat_listesi = sorted(
            hi[hi["toplam_sefer"] >= 50]["HATKODU"].dropna().unique().tolist()
        )
        sc1, _ = st.columns([1, 3])
        with sc1:
            secili = st.selectbox("Aksayan hattı seçin:", hat_listesi)

        # Önce hat_onerileri tanımla
        hat_onerileri     = oneri[oneri["aksayan_hat"] == secili].copy()
        alternatif_hatlar = hat_onerileri["alternatif"].tolist()
        hat_row           = hi[hi["HATKODU"] == secili]

        # Etkilenen hat — dom veya oneri CSV'den al
        etkilenen_n = 0
        if dom is not None:
            dom_match = dom[
                (dom["hat_adi"] == secili) |
                (dom["hat_id"]  == secili)
            ]
            if len(dom_match) > 0:
                etkilenen_n = int(dom_match["etkilenen_hat"].values[0])
        if etkilenen_n == 0 and len(hat_onerileri) > 0:
            etkilenen_n = int(hat_onerileri["etkilened_hat"].iloc[0]) \
                          if "etkilened_hat" in hat_onerileri.columns \
                          else int(hat_onerileri["etkilenen_hat"].iloc[0])

        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi-card" style="border-top:3px solid #DC2626"><div class="kpi-val" style="color:#DC2626">{etkilenen_n}</div><div class="kpi-label">Etkilenen Hat</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-card" style="border-top:3px solid #2563EB"><div class="kpi-val" style="color:#2563EB">{len(hat_onerileri)}</div><div class="kpi-label">Alternatif Hat Önerisi</div></div>', unsafe_allow_html=True)
        iptal = f"%{hat_row['IPTAL_ORANI'].values[0]*100:.1f}" if len(hat_row) > 0 else "—"
        k3.markdown(f'<div class="kpi-card" style="border-top:3px solid #CA8A04"><div class="kpi-val" style="color:#CA8A04">{iptal}</div><div class="kpi-label">Hattın İptal Oranı</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── İstanbul Haritası ─────────────────────────────
        if guzergah_df is not None:
            st.markdown('<div class="section-title">İstanbul Hat Haritası</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">'
                '🗺️ <b style="color:#DC2626">Kırmızı:</b> Aksayan hat &nbsp;|&nbsp; '
                '<b style="color:#16A34A">Yeşil:</b> Önerilen alternatifler &nbsp;|&nbsp; '
                '<b style="color:#94A3B8">Gri:</b> Diğer hatlar'
                '</div>',
                unsafe_allow_html=True
            )

            harita_fig = go.Figure()

            # Tüm hatlar — gri arka plan
            for _, row in guzergah_df.iterrows():
                hat   = row['hat_kodu']
                koord = row['koordinatlar']
                if not koord or hat == secili or hat in alternatif_hatlar:
                    continue
                lats = [k[0] for k in koord]
                lons = [k[1] for k in koord]
                harita_fig.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons,
                    mode='lines',
                    line=dict(width=1, color='rgba(148,163,184,0.3)'),
                    hoverinfo='skip',
                    showlegend=False,
                ))

            # Alternatif hatlar — yeşil
            for i, alt_hat in enumerate(alternatif_hatlar):
                alt_row = guzergah_df[guzergah_df['hat_kodu'] == alt_hat]
                if len(alt_row) == 0:
                    continue
                koord = alt_row.iloc[0]['koordinatlar']
                if not koord:
                    continue
                lats = [k[0] for k in koord]
                lons = [k[1] for k in koord]
                harita_fig.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons,
                    mode='lines',
                    line=dict(width=4, color='#16A34A'),
                    name=f'#{i+1} Alternatif: {alt_hat}',
                    hovertemplate=f'<b>#{i+1} Alternatif Hat: {alt_hat}</b><extra></extra>',
                ))

            # Aksayan hat — kırmızı (en üstte)
            merkez_lat, merkez_lon = 41.015, 28.979
            aksayan_row = guzergah_df[guzergah_df['hat_kodu'] == secili]
            if len(aksayan_row) > 0:
                koord = aksayan_row.iloc[0]['koordinatlar']
                if koord:
                    lats = [k[0] for k in koord]
                    lons = [k[1] for k in koord]
                    harita_fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons,
                        mode='lines+markers',
                        line=dict(width=6, color='#DC2626'),
                        marker=dict(size=5, color='#DC2626'),
                        name=f'Aksayan: {secili}',
                        hovertemplate=f'<b>Aksayan Hat: {secili}</b><extra></extra>',
                    ))
                    merkez_lat = sum(lats) / len(lats)
                    merkez_lon = sum(lons) / len(lons)

            harita_fig.update_layout(
                mapbox=dict(
                    style="carto-positron",
                    center=dict(lat=merkez_lat, lon=merkez_lon),
                    zoom=11,
                ),
                height=520,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E9ECEF",
                    borderwidth=1,
                    font=dict(size=11),
                    x=0.01, y=0.99,
                ),
            )
            st.plotly_chart(harita_fig, use_container_width=True)

        # ── Alternatif öneriler ───────────────────────────
        if len(hat_onerileri) > 0:
            st.markdown(
                f'<div class="section-title">✅ {secili} Aksarsa — Önerilen Alternatif Hatlar</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="success-box">Aşağıdaki hatlar <b>ortak durak sayısı</b>, '
                '<b>güvenilirlik</b> ve <b>coğrafi yakınlık</b> skoruna göre sıralanmıştır.</div>',
                unsafe_allow_html=True)

            for _, r in hat_onerileri.iterrows():
                iptal_renk = ("#16A34A" if r["iptal_orani"] < 10
                              else "#CA8A04" if r["iptal_orani"] < 15
                              else "#DC2626")
                skor_pct = int(r["skor"] * 100)
                st.markdown(f"""
                <div class="oneri-card">
                  <div style="display:flex;justify-content:space-between;
                       align-items:center;margin-bottom:10px;">
                    <span style="font-size:18px;font-weight:700;color:#1A1A2E;">
                      Hat {r['alternatif']}</span>
                    <span style="background:#EFF6FF;color:#1D4ED8;
                          padding:3px 12px;border-radius:20px;
                          font-size:12px;font-weight:600;">
                      #{int(r['oneri_sira'])}. Öneri</span>
                  </div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                       gap:12px;font-size:13px;margin-bottom:10px;">
                    <div><span style="color:#6C757D">Konum</span><br>
                         <b>{r['konum']}</b></div>
                    <div><span style="color:#6C757D">Ortak Durak</span><br>
                         <b>{int(r['ortak_durak'])} durak</b></div>
                    <div><span style="color:#6C757D">İptal Oranı</span><br>
                         <b style="color:{iptal_renk}">%{r['iptal_orani']}</b></div>
                  </div>
                  <div style="font-size:11px;color:#6C757D;margin-bottom:4px;">
                    Uygunluk Skoru: %{skor_pct}</div>
                  <div style="background:#F1F5F9;border-radius:4px;
                       height:6px;overflow:hidden;">
                    <div style="width:{skor_pct}%;background:#2563EB;
                         height:100%;border-radius:4px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="warn-box">⚠️ <b>{secili}</b> için uygun alternatif hat bulunamadı.</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── Genel domino grafiği ──────────────────────────────
    if dom is not None:
        st.markdown('<div class="section-title">Hat Kapanırsa Kaç Hat Etkilenir?</div>', unsafe_allow_html=True)
        fig = px.bar(
            dom.sort_values("etkilenen_hat", ascending=False),
            x="hat_adi", y="etkilenen_hat",
            color="etkilenen_hat",
            color_continuous_scale=["#FEE2E2","#DC2626"],
            text="etkilenen_hat",
            labels={"hat_adi":"Hat","etkilenen_hat":"Etkilenen Hat Sayısı"})
        fig.update_traces(textposition="outside")
        fig.update_layout(**LIGHT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if khat is not None:
        st.markdown('<div class="section-title">Sistem İçin En Kritik Hatlar</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">📐 <b>Betweenness Centrality</b> — bir hattın diğer hatlar arasında ne kadar "köprü" görevi gördüğünü ölçer. Yüksek değer = o hat kapanınca ağ en çok zarar görür.</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            khat.sort_values("betweenness").tail(15),
            x="betweenness", y="hat_adi", orientation="h",
            color="derece",
            color_continuous_scale=["#DBEAFE","#1D4ED8"],
            labels={"betweenness":"Betweenness","hat_adi":"Hat","derece":"Bağlantı Sayısı"},
            text="derece")
        fig2.update_traces(texttemplate="%{text} hat", textposition="outside")
        fig2.update_layout(**LIGHT, height=500, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    if hi is not None:
        st.markdown('<div class="section-title">En Yüksek İptal Oranına Sahip Hatlar</div>', unsafe_allow_html=True)
        hi_f = (hi[hi["toplam_sefer"] >= 100]
                .sort_values("IPTAL_ORANI", ascending=False).head(20))
        fig3 = px.bar(
            hi_f, x="HATKODU", y="IPTAL_ORANI",
            color="IPTAL_ORANI",
            color_continuous_scale=["#FEF9C3","#DC2626"],
            labels={"HATKODU":"Hat","IPTAL_ORANI":"İptal Oranı"},
            text=hi_f["IPTAL_ORANI"].apply(lambda x: f"%{x*100:.1f}"))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(**LIGHT, height=420, coloraxis_showscale=False)
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
# ════════════════════════════════════════════════════════════
# 🗺️ KRİZ HARİTASI
# ════════════════════════════════════════════════════════════
elif "Kriz Haritası" in sayfa:
    st.markdown("## 🗺️ İstanbul İlçe Bazlı Kriz Haritası")
    st.markdown('<div class="info-box">📍 İlçe bazında kaza sayısı ve kriz tetikleyici oranı analiz edildi. <b>Kriz oranı:</b> ölümlü kaza, 3+ yaralı veya kötü hava koşulu içeren olayların tüm kazalara oranı.</div>', unsafe_allow_html=True)

    ilce = V["ilce_risk"]
    if ilce is not None:
        ilce_f = ilce[ilce["ILCE"] != "BELİRSİZ"].copy()

        st.markdown('<div class="section-title">İlçe Bazında Kaza Sayısı (Renk = Kriz Oranı)</div>', unsafe_allow_html=True)
        fig1 = px.bar(ilce_f.sort_values("kaza_sayisi", ascending=False).head(15),
                      x="ILCE", y="kaza_sayisi",
                      color="kriz_orani",
                      color_continuous_scale=["#DCFCE7","#DC2626"],
                      labels={"ILCE":"İlçe","kaza_sayisi":"Kaza Sayısı",
                              "kriz_orani":"Kriz Oranı"},
                      text="kaza_sayisi")
        fig1.update_traces(textposition="outside")
        fig1.update_layout(**LIGHT, height=460, coloraxis_showscale=True,
                           coloraxis_colorbar=dict(title="Kriz Oranı"))
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown('<div class="section-title">Kaza Yoğunluğu × Kriz Oranı (Baloncuk = Yaralı Sayısı)</div>', unsafe_allow_html=True)
        fig2 = px.scatter(ilce_f,
                          x="kaza_sayisi", y="kriz_orani",
                          size="toplam_yarali",
                          color="kriz_orani",
                          color_continuous_scale=["#DCFCE7","#DC2626"],
                          hover_name="ILCE", text="ILCE",
                          labels={"kaza_sayisi":"Kaza Sayısı",
                                  "kriz_orani":"Kriz Oranı",
                                  "toplam_yarali":"Yaralı"})
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.update_layout(**LIGHT, height=480, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">Tüm İlçeler — Detay Tablosu</div>', unsafe_allow_html=True)
        st.dataframe(ilce_f.sort_values("kaza_sayisi", ascending=False)
                           .reset_index(drop=True),
                     use_container_width=True, height=320)


# ════════════════════════════════════════════════════════════
# 📈 TREND ANALİZİ
# ════════════════════════════════════════════════════════════
elif "Trend" in sayfa:
    st.markdown("## 📈 Arıza & Kaza Zaman Serisi")
    st.markdown('<div class="info-box">📅 2024 ve 2025 yılları aylık bazda karşılaştırıldı. Günlük sefer verisi üzerinde <b>Z-Score anomali tespiti</b> uygulandı — kırmızı yıldızlar anormal yoğunluk yaşanan günleri gösterir.</div>', unsafe_allow_html=True)

    aa = V["aylik_ariza"]
    ak = V["aylik_kaza"]
    g  = V["gunluk"]

    if aa is not None:
        aa["TARIH"] = pd.to_datetime(aa["TARIH"])
        st.markdown('<div class="section-title">Aylık Arıza — 2024 vs 2025</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        for yil, renk, dash in [(2024,"#2563EB","solid"),(2025,"#DC2626","solid")]:
            d = aa[aa["YIL"] == yil]
            fig1.add_trace(go.Scatter(
                x=d["TARIH"], y=d["ariza_sayisi"],
                name=str(yil), line=dict(color=renk, width=2.5, dash=dash),
                mode="lines+markers", marker=dict(size=7)))
        fig1.update_layout(**LIGHT, height=400,
                           yaxis_title="Arıza Sayısı",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig1, use_container_width=True)

    if ak is not None:
        ak["TARIH"] = pd.to_datetime(ak["TARIH"])
        st.markdown('<div class="section-title">Aylık Kaza — 2024 vs 2025</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for yil, renk in [(2024,"#16A34A"),(2025,"#EA580C")]:
            d = ak[ak["KAZA_YIL"] == yil]
            fig2.add_trace(go.Scatter(
                x=d["TARIH"], y=d["kaza_sayisi"],
                name=str(yil), line=dict(color=renk, width=2.5),
                mode="lines+markers", marker=dict(size=7)))
        fig2.update_layout(**LIGHT, height=400,
                           yaxis_title="Kaza Sayısı",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

    if g is not None:
        g["TARIH"] = pd.to_datetime(g["TARIH"])
        st.markdown('<div class="section-title">Günlük Sefer Anomalileri (Z-Score > 2)</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=g["TARIH"], y=g["sefer_sayisi"],
            mode="lines", name="Günlük Sefer",
            line=dict(color="#2563EB", width=1.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)"))
        an = g[g["SEFER_ANOMALI"] == True]
        fig3.add_trace(go.Scatter(
            x=an["TARIH"], y=an["sefer_sayisi"],
            mode="markers", name=f"Anomali ({len(an)} gün)",
            marker=dict(color="#DC2626", size=11,
                        symbol="star", line=dict(width=1, color="#fff"))))
        fig3.update_layout(**LIGHT, height=420,
                           yaxis_title="Sefer Sayısı",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════
# ⚠️ ERKEN UYARI
# ════════════════════════════════════════════════════════════
elif "Erken Uyarı" in sayfa:
    st.markdown("## ⚠️ Denetim → Arıza Erken Uyarı")
    st.markdown('<div class="info-box">🔍 Denetim verisi ile arıza arasında güçlü bir örüntü tespit edildi. <b>Olumsuz denetimden sonra araçların %71.3\'ü 14 gün içinde arıza yaptı.</b> Bu 14 günlük pencere proaktif bakım için kritik bir fırsat sunuyor.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Korelasyon (r)",        "0.545",  help="Denetim olumsuzluk oranı × arıza sayısı")
    c2.metric("14 Gün İçi Arıza",      "%71.3",  help="Olumsuz denetim sonrası 14 gün penceresi")
    c3.metric("Ort. Olumsuz Denetim",  "%38.2",  help="Araç başına ortalama olumsuz denetim oranı")
    c4.metric("Birliktelik Kuralı",    "380",    help="Arıza birliktelik kuralı sayısı")
    st.divider()

    zay = V["zayi"]
    den = V["denetim"]
    bir = V["birliktelik"]

    if zay is not None:
        st.markdown('<div class="section-title">Sefer İptaline Yol Açan Arıza Türleri</div>', unsafe_allow_html=True)
        st.markdown('<div class="warn-box">⚡ Bu arızalar hem araç arızası hem de sefer iptali yaratıyor — <b>çift operasyonel zarar.</b> Öncelikli bakım planlamasında bu türler dikkate alınmalı.</div>', unsafe_allow_html=True)
        zay_r = zay.reset_index() if "ARIZAUSTKODTANIM" not in zay.columns else zay
        zay_r = zay_r.sort_values("toplam_zayi_sefer", ascending=True).tail(10)
        fig1 = px.bar(zay_r,
                      x="toplam_zayi_sefer", y="ARIZAUSTKODTANIM",
                      orientation="h",
                      color="ort_zayi_sefer",
                      color_continuous_scale=["#FEF9C3","#DC2626"],
                      labels={"ARIZAUSTKODTANIM":"Arıza Türü",
                              "toplam_zayi_sefer":"Toplam Zayi Sefer",
                              "ort_zayi_sefer":"Ort. Zayi/Arıza"},
                      text="toplam_zayi_sefer")
        fig1.update_traces(textposition="outside")
        fig1.update_layout(**LIGHT, height=460, coloraxis_showscale=True,
                           coloraxis_colorbar=dict(title="Ort. Zayi"))
        st.plotly_chart(fig1, use_container_width=True)

    if den is not None:
        st.markdown('<div class="section-title">Araç Başına Olumsuz Denetim Oranı</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">📊 X ekseni: olumsuz denetim oranı (0=hiç olumsuz yok, 1=hep olumsuz). <b>Sağa kaymış araçlar yüksek riskli</b> — bunlar RF modelinde de kritik çıktı.</div>', unsafe_allow_html=True)
        den_f = den[den["toplam_denetim"] >= 5]
        fig2 = px.histogram(den_f, x="OLUMSUZ_ORAN", nbins=40,
                            color_discrete_sequence=["#2563EB"],
                            labels={"OLUMSUZ_ORAN":"Olumsuz Denetim Oranı",
                                    "count":"Araç Sayısı"})
        fig2.update_layout(**LIGHT, height=380,
                           xaxis_title="Olumsuz Denetim Oranı",
                           yaxis_title="Araç Sayısı")
        st.plotly_chart(fig2, use_container_width=True)

    if bir is not None:
        st.markdown('<div class="section-title">Arıza Birliktelik Kuralları</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">🔗 <b>Lift > 1</b> demek: bu iki arıza birlikte görülme ihtimali rastgele beklentinin üzerinde. <b>Lift yükseldikçe ilişki güçlenir</b> — bir arızayı görünce diğerine karşı önceden hazırlık yapılabilir.</div>', unsafe_allow_html=True)
        bir_f = (bir[bir["lift"] > 1.2]
                 .sort_values("lift", ascending=False)
                 .head(15).reset_index(drop=True))
        st.dataframe(bir_f, use_container_width=True, height=380)