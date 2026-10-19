# -*- coding: utf-8 -*-
"""Paleolatitude and paleomagnetic-pole plotting app.

Run with:
    python -m streamlit run paleo-latidude.py

Optional vector export dependency:
    python -m pip install kaleido
"""
from __future__ import annotations

import hashlib
import io
import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Paleolatitude & Paleomagnetic Poles",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* A restrained scientific-app theme: polish the existing layout without changing its workflow. */
:root {
  --pm-ink: #183047;
  --pm-muted: #617386;
  --pm-accent: #176b87;
  --pm-accent-light: #eaf4f7;
  --pm-border: #dce5eb;
}
.stApp { background: linear-gradient(180deg, #f7fafc 0%, #ffffff 260px); }
.block-container { padding-top: 1.55rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { color: var(--pm-ink) !important; letter-spacing: -0.025em; }
h1 { font-size: 2.05rem !important; font-weight: 760 !important; margin-bottom: .2rem !important; }
h2 { font-size: 1.45rem !important; font-weight: 720 !important; padding-bottom: .42rem; border-bottom: 1px solid var(--pm-border); margin-top: 1.4rem !important; }
h3 { font-size: 1.08rem !important; font-weight: 680 !important; }
[data-testid="stCaptionContainer"] { color: var(--pm-muted); }
[data-testid="stSidebar"] { background: #f1f6f9; border-right: 1px solid var(--pm-border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--pm-ink) !important; }
[data-testid="stFileUploader"] { background: #ffffff; border: 1px dashed #b8cbd6; border-radius: 12px; padding: .45rem .65rem; }
[data-testid="stDownloadButton"] button, [data-testid="stButton"] button { border-radius: 9px; border: 1px solid #bfd1dc; font-weight: 620; transition: all .15s ease; }
[data-testid="stDownloadButton"] button:hover, [data-testid="stButton"] button:hover { border-color: var(--pm-accent); color: var(--pm-accent); }
[data-testid="stDataFrame"] { border: 1px solid var(--pm-border); border-radius: 10px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 10px; }
hr { border-color: var(--pm-border); }
</style>
""", unsafe_allow_html=True)
st.title("Paleolatitude and Paleomagnetic Pole Analysis")
st.caption("APWP paleolatitude curves · Schmidt equal-area stereonet · equal-paleolatitude small-circle comparison")
st.markdown("<div style='height:4px;background:linear-gradient(90deg,#176b87,#76b7c5,#dcecf0);border-radius:6px;margin:0 0 1.15rem 0'></div>", unsafe_allow_html=True)

# ----------------------------- Utilities -----------------------------
def norm_lon(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def paleolatitude(pole_lat, pole_lon, ref_lat, ref_lon):
    """Signed paleolatitude using the angular distance to the supplied pole."""
    plat = np.radians(np.asarray(pole_lat, dtype=float))
    plon = np.radians(np.asarray(pole_lon, dtype=float))
    rlat = math.radians(float(ref_lat))
    rlon = math.radians(float(ref_lon))
    cos_d = np.sin(rlat) * np.sin(plat) + np.cos(rlat) * np.cos(plat) * np.cos(plon - rlon)
    d = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    return 90.0 - d


def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    c = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def antipode(lat, lon):
    return -np.asarray(lat, dtype=float), norm_lon(np.asarray(lon, dtype=float) + 180.0)


def read_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(raw))
    # Try normal CSV parsing first, then delimiter autodetection.
    try:
        df = pd.read_csv(io.BytesIO(raw))
        if df.shape[1] == 1:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine='python')
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine='python')


def clean_colname(c):
    return str(c).strip().lower().replace(' ', '_').replace('-', '_')


def make_unique_columns(columns):
    """Normalize duplicate headers so df['field'] always returns one Series."""
    used = {}
    result = []
    for col in columns:
        name = clean_colname(col)
        count = used.get(name, 0) + 1
        used[name] = count
        result.append(name if count == 1 else f'{name}__duplicate_{count}')
    return result


def standardize_apwp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # If pandas interpreted a headerless numeric first row as column names, restore it.
    if len(df.columns) >= 4 and all(pd.to_numeric(str(c), errors='coerce') == pd.to_numeric(str(c), errors='coerce') for c in df.columns[:4]):
        first = pd.DataFrame([[pd.to_numeric(str(c), errors='coerce') for c in df.columns]], columns=df.columns)
        df = pd.concat([first, df], ignore_index=True)
        df.columns = [f'col_{i+1}' for i in range(len(df.columns))]
    df.columns = make_unique_columns(df.columns)
    aliases = {
        'longitude': ['longitude', 'lon', 'plon', 'pole_lon', 'pole_longitude', 'long'],
        'latitude': ['latitude', 'lat', 'plat', 'pole_lat', 'pole_latitude'],
        'age': ['age', 'age_ma', 'ma', 'age_(ma)', 'time'],
        'a95': ['a95', 'a_95', 'alpha95', 'alpha_95', 'dp', 'error'],
        'min_age': ['min_age', 'age_min', 'minimum_age', 'young_age'],
        'max_age': ['max_age', 'age_max', 'maximum_age', 'old_age'],
        'age_error': ['age_error', 'age_err', 'age_uncertainty', 'sigma_age'],
        'label': ['label', 'name', 'site', 'formation', 'reference', 'study'],
    }
    rename = {}
    for target, options in aliases.items():
        if target not in df.columns:
            for col in options:
                if col in df.columns:
                    rename[col] = target
                    break
    df = df.rename(columns=rename)
    # Common four-column headerless APWP: longitude, latitude, A95, age.
    if not {'longitude', 'latitude', 'age'}.issubset(df.columns) and df.shape[1] >= 4:
        # Only use positional fallback if the three essential fields cannot be identified.
        df = df.iloc[:, :max(4, min(df.shape[1], 8))].copy()
        cols = list(df.columns)
        positional = {cols[0]: 'longitude', cols[1]: 'latitude', cols[2]: 'a95', cols[3]: 'age'}
        df = df.rename(columns=positional)
    for c in ['longitude', 'latitude', 'age', 'a95', 'min_age', 'max_age', 'age_error']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    required = {'longitude', 'latitude', 'age'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('APWP 表格无法识别必要字段：' + ', '.join(sorted(missing)) + '。至少需要 longitude、latitude、age。')
    df = df.dropna(subset=['longitude', 'latitude', 'age'])
    df['longitude'] = norm_lon(df['longitude'])
    df['latitude'] = df['latitude'].clip(-90, 90)
    if 'label' not in df.columns:
        df['label'] = [f'APWP {i+1}' for i in range(len(df))]
    return df.reset_index(drop=True)


def standardize_database(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # If pandas interpreted a headerless numeric first row as column names, restore it.
    if len(df.columns) >= 3 and all(pd.to_numeric(str(c), errors='coerce') == pd.to_numeric(str(c), errors='coerce') for c in df.columns[:3]):
        first = pd.DataFrame([[pd.to_numeric(str(c), errors='coerce') for c in df.columns]], columns=df.columns)
        df = pd.concat([first, df], ignore_index=True)
        df.columns = [f'col_{i+1}' for i in range(len(df.columns))]
    df.columns = make_unique_columns(df.columns)
    aliases = {
        'pole_lat': ['pole_lat', 'plat', 'latitude', 'lat', 'pole_latitude', 'p_lat'],
        'pole_lon': ['pole_lon', 'plon', 'longitude', 'lon', 'pole_longitude', 'p_lon'],
        'age': ['age', 'age_ma', 'ma', 'age_(ma)', 'mean_age', 'age_mid'],
        'a95': ['a95', 'a_95', 'alpha95', 'alpha_95'],
        'min_age': ['min_age', 'age_min', 'minimum_age', 'young_age'],
        'max_age': ['max_age', 'age_max', 'maximum_age', 'old_age'],
        'age_error': ['age_error', 'age_err', 'age_uncertainty', 'sigma_age'],
        'name': ['name', 'site', 'formation', 'unit', 'locality', 'reference', 'study', 'sample'],
    }
    rename = {}
    for target, options in aliases.items():
        if target not in df.columns:
            for col in options:
                if col in df.columns:
                    rename[col] = target
                    break
    df = df.rename(columns=rename)
    if not {'pole_lat', 'pole_lon', 'age'}.issubset(df.columns) and df.shape[1] >= 3:
        cols = list(df.columns)
        # Historical database convention: first columns plat, plon, age.
        df = df.rename(columns={cols[0]: 'pole_lat', cols[1]: 'pole_lon', cols[2]: 'age'})
    for c in ['pole_lat', 'pole_lon', 'age', 'a95', 'min_age', 'max_age', 'age_error']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    required = {'pole_lat', 'pole_lon', 'age'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Research database 无法识别必要字段：' + ', '.join(sorted(missing)) + '。至少需要 plat/pole_lat、plon/pole_lon 和 age。')
    df = df.dropna(subset=['pole_lat', 'pole_lon', 'age'])
    df['pole_lat'] = df['pole_lat'].clip(-90, 90)
    df['pole_lon'] = norm_lon(df['pole_lon'])
    if 'name' not in df.columns:
        df['name'] = [f'Database {i+1}' for i in range(len(df))]
    df['name'] = df['name'].fillna('').astype(str)
    return df.reset_index(drop=True)


def get_age_bounds(df):
    if 'min_age' in df.columns and 'max_age' in df.columns:
        lo = df['min_age'].where(df['min_age'].notna(), df['age'])
        hi = df['max_age'].where(df['max_age'].notna(), df['age'])
    elif 'age_error' in df.columns:
        err = df['age_error'].fillna(0).abs()
        lo, hi = df['age'] - err, df['age'] + err
    else:
        lo = hi = df['age']
    return lo.astype(float), hi.astype(float)


def vector_export(fig, base_name, key):
    c1, c2 = st.columns(2)
    for col, fmt, mime, label in [
        (c1, 'svg', 'image/svg+xml', '下载 SVG（可编辑矢量图）'),
        (c2, 'pdf', 'application/pdf', '下载 PDF（矢量格式）'),
    ]:
        with col:
            try:
                payload = fig.to_image(format=fmt, engine='kaleido', scale=2)
                st.download_button(label, data=payload, file_name=f'{base_name}.{fmt}', mime=mime, key=f'{key}_{fmt}')
            except Exception as exc:
                st.download_button(label + '（不可用）', data=f'导出失败：{exc}'.encode('utf-8'), file_name=f'{base_name}_{fmt}_export_error.txt', mime='text/plain', key=f'{key}_{fmt}_error')
                st.caption('矢量导出需要安装兼容版本的 kaleido：`python -m pip install -U kaleido`。')


def polish_figure(fig, *, height=None):
    """Apply a consistent, publication-friendly style to Plotly figures."""
    layout = dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, Helvetica, sans-serif", size=12, color="#263746"),
        title=dict(font=dict(size=17, color="#183047"), x=0.02, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#e0e8ed",
            borderwidth=0.6,
            font=dict(size=11),
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor="#183047", bordercolor="#183047", font=dict(color="#ffffff", size=12)),
        margin=dict(l=58, r=28, t=68, b=58),
    )
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    return fig


def add_circle(fig, radius_deg, name, color, ref_lat, ref_lon, n=361):
    """Add equal-angular-distance small circle centered on the geographic reference point."""
    if radius_deg < 1e-8:
        return
    bearings = np.linspace(0, 360, n)
    lat1, lon1, d = np.radians([ref_lat, ref_lon, radius_deg])
    br = np.radians(bearings)
    lat2 = np.arcsin(np.clip(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(br), -1, 1))
    lon2 = lon1 + np.arctan2(np.sin(br)*np.sin(d)*np.cos(lat1), np.cos(d)-np.sin(lat1)*np.sin(lat2))
    lats, lons = np.degrees(lat2), np.degrees(lon2)
    xs, ys = [], []
    # Fold antipodal hemispheres onto the upper hemisphere, breaking lines at the fold.
    prev_hemi = None
    for lat, lon in zip(lats, lons):
        hemi = lat >= 0
        if prev_hemi is not None and hemi != prev_hemi:
            xs.append(None); ys.append(None)
        if not hemi:
            lat, lon = antipode(lat, lon)
        r = math.sqrt(2.0) * math.sin(math.radians(90.0 - abs(float(lat))) / 2.0)
        xs.append(r * math.sin(math.radians(float(lon))))
        ys.append(r * math.cos(math.radians(float(lon))))
        prev_hemi = hemi
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name, line=dict(color=color, width=2, dash='dash')))


def add_confidence_circle(fig, pole_lat, pole_lon, a95, name, color, showlegend=False, n=181):
    """Draw an A95 confidence cone boundary around a pole on the Schmidt net.

    A95 is treated as an angular radius on the sphere. Southern-hemisphere
    poles are folded to their antipodes for display, matching the main net.
    """
    if pd.isna(a95) or not np.isfinite(float(a95)) or float(a95) <= 0:
        return
    radius_deg = min(float(a95), 180.0)
    # Fold the centre into the upper hemisphere before tracing its confidence cone.
    center_lat, center_lon = float(pole_lat), float(pole_lon)
    if center_lat < 0:
        center_lat, center_lon = float(antipode(center_lat, center_lon)[0]), float(antipode(center_lat, center_lon)[1])

    lat1, lon1, delta = np.radians([center_lat, center_lon, radius_deg])
    bearings = np.radians(np.linspace(0.0, 360.0, n))
    lat2 = np.arcsin(np.clip(
        np.sin(lat1) * np.cos(delta) +
        np.cos(lat1) * np.sin(delta) * np.cos(bearings), -1.0, 1.0
    ))
    lon2 = lon1 + np.arctan2(
        np.sin(bearings) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2)
    )
    circle_lats, circle_lons = np.degrees(lat2), np.degrees(lon2)
    xs, ys = [], []
    previous_hemi = None
    for lat, lon in zip(circle_lats, circle_lons):
        is_north = lat >= 0
        if previous_hemi is not None and is_north != previous_hemi:
            # Do not draw a false straight segment across the net when the circle
            # crosses the equator and is folded to the antipodal hemisphere.
            xs.append(None)
            ys.append(None)
        if not is_north:
            lat, lon = float(antipode(lat, lon)[0]), float(antipode(lat, lon)[1])
        r = math.sqrt(2.0) * math.sin(math.radians(90.0 - abs(float(lat))) / 2.0)
        xs.append(r * math.sin(math.radians(float(lon))))
        ys.append(r * math.cos(math.radians(float(lon))))
        previous_hemi = is_north
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines', name=name,
        line=dict(color=color, width=1.1),
        opacity=0.72, hovertemplate=f'{name}<br>A95 = {radius_deg:.2f}°<extra></extra>',
        showlegend=showlegend, legendgroup=name
    ))


def stereonet_figure(points, selected_indices=None, ref_lat=None, ref_lon=None, circle_rows=None):
    fig = go.Figure()
    # Equal-area Schmidt net boundary and latitude circles.
    theta = np.linspace(0, 2*np.pi, 721)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines', line=dict(color='#222', width=1.5), name='Net boundary', hoverinfo='skip', showlegend=False))
    for lat in [0, 15, 30, 45, 60, 75]:
        r = math.sqrt(2.0) * math.sin(math.radians(90-lat)/2.0)
        fig.add_trace(go.Scatter(x=r*np.sin(theta), y=r*np.cos(theta), mode='lines', line=dict(color='#c8c8c8', width=0.7), hoverinfo='skip', showlegend=False, name=f'{lat}° latitude'))
    for lon in range(0, 360, 30):
        rr = np.linspace(0, 1, 100)
        rad = math.radians(lon)
        fig.add_trace(go.Scatter(x=rr*np.sin(rad), y=rr*np.cos(rad), mode='lines', line=dict(color='#dedede', width=0.7), hoverinfo='skip', showlegend=False))
    if len(points):
        # Plot A95 confidence-circle boundaries before the pole symbols so the
        # symbols remain visible. Missing A95 values are intentionally skipped.
        legend_seen = set()
        for _, row in points.iterrows():
            if 'a95' not in row.index or pd.isna(row.get('a95')):
                continue
            original_lat = float(row['pole_lat'])
            original_hemi = 'N' if original_lat >= 0 else 'S'
            circle_color = '#d62728' if original_hemi == 'N' else '#1f77b4'
            legend_name = f'A95 confidence circle ({original_hemi} pole)'
            add_confidence_circle(
                fig, row['pole_lat'], row['pole_lon'], row.get('a95'),
                legend_name, circle_color, showlegend=legend_name not in legend_seen
            )
            legend_seen.add(legend_name)
        xs, ys, labels, custom, colors, symbols = [], [], [], [], [], []
        for _, row in points.iterrows():
            lat, lon = float(row['pole_lat']), float(row['pole_lon'])
            original_hemi = 'N' if lat >= 0 else 'S'
            if lat < 0:
                lat, lon = antipode(lat, lon)
            r = math.sqrt(2.0) * math.sin(math.radians(90.0-abs(lat))/2.0)
            xs.append(r*math.sin(math.radians(lon)))
            ys.append(r*math.cos(math.radians(lon)))
            labels.append(str(row.get('label', row.get('name', 'pole'))))
            custom.append([float(row['pole_lat']), float(row['pole_lon']), float(row['age']), original_hemi, float(row.get('paleolatitude', np.nan))])
            colors.append('#d62728' if original_hemi == 'N' else '#1f77b4')
            symbols.append('circle' if original_hemi == 'N' else 'diamond')
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='Paleomagnetic poles', text=labels,
            customdata=custom, marker=dict(size=9, color=colors, symbol=symbols, line=dict(color='white', width=0.7)),
            hovertemplate='<b>%{text}</b><br>Age: %{customdata[2]:.2f} Ma<br>Original pole: %{customdata[0]:.2f}°, %{customdata[1]:.2f}°<br>Original hemisphere: %{customdata[3]}<br>Paleolatitude: %{customdata[4]:.2f}°<extra></extra>'))
    if ref_lat is not None and ref_lon is not None:
        lat, lon = float(ref_lat), float(ref_lon)
        if lat < 0:
            lat, lon = antipode(lat, lon)
        r = math.sqrt(2.0)*math.sin(math.radians(90.0-abs(lat))/2.0)
        x, y = r*math.sin(math.radians(lon)), r*math.cos(math.radians(lon))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name='Reference point (folded to upper hemisphere)', marker=dict(size=13, symbol='x', color='black'), hovertemplate=f'Reference point<br>{ref_lat:.2f}°, {ref_lon:.2f}°<extra></extra>'))
    if circle_rows and ref_lat is not None and ref_lon is not None:
        for row, label, color in circle_rows:
            d = 90.0 - float(paleolatitude(row['pole_lat'], row['pole_lon'], ref_lat, ref_lon))
            add_circle(fig, d, label, color, ref_lat, ref_lon)
    fig.update_layout(title='Paleomagnetic poles — Schmidt equal-area stereonet',
        xaxis=dict(visible=False, range=[-1.08, 1.08], scaleanchor='y', scaleratio=1, zeroline=False),
        yaxis=dict(visible=False, range=[-1.08, 1.08], zeroline=False),
        height=700, margin=dict(l=25,r=25,t=65,b=70),
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'))
    polish_figure(fig, height=700)
    return fig

# ----------------------------- Sidebar -----------------------------
st.sidebar.header('分析设置')
ref_lat = st.sidebar.number_input('参考点纬度 (°)', value=36.27, min_value=-90.0, max_value=90.0, step=0.01, format='%.2f')
ref_lon = st.sidebar.number_input('参考点经度 (°)', value=98.64, min_value=-180.0, max_value=180.0, step=0.01, format='%.2f')
st.sidebar.caption('古纬度按 λ = 90° − 球面角距计算；请确认输入极点的极性/半球约定。')

apwp_uploads = st.file_uploader('上传 APWP 数据（可一次多选 CSV / Excel）', type=['csv','txt','xlsx','xls'], accept_multiple_files=True, key='apwp_uploads')
db_upload = st.file_uploader('上传 Research database（CSV / Excel）', type=['csv','txt','xlsx','xls'], key='db_upload')

apwp = pd.DataFrame()
database = pd.DataFrame()
if apwp_uploads:
    apwp_frames = []
    for uploaded_file in apwp_uploads:
        try:
            frame = standardize_apwp(read_table(uploaded_file))
            frame['source_file'] = uploaded_file.name
            # Keep curves separate and make labels unambiguous when files are combined.
            frame['label'] = frame['label'].astype(str)
            frame['display_label'] = uploaded_file.name + ' | ' + frame['label']
            apwp_frames.append(frame)
            st.success(f'已载入 APWP：{uploaded_file.name}（{len(frame)} 行）')
        except Exception as exc:
            st.error(f'APWP 文件读取失败（{uploaded_file.name}）：{exc}')
    if apwp_frames:
        apwp = pd.concat(apwp_frames, ignore_index=True, sort=False)
        apwp['paleolatitude'] = paleolatitude(apwp['latitude'], apwp['longitude'], ref_lat, ref_lon)
        apwp['label'] = apwp['display_label']
        st.info(f'APWP 合计：{len(apwp_frames)} 个文件，{len(apwp)} 行数据。绘图时将按文件分别显示曲线。')
if db_upload is not None:
    try:
        incoming = standardize_database(read_table(db_upload))
        fingerprint = hashlib.sha256(db_upload.getvalue()).hexdigest()[:16]
        state_key = f'database_working_{fingerprint}'
        if state_key not in st.session_state:
            incoming['_record_id'] = [f'DB-{i+1:04d}' for i in range(len(incoming))]
            incoming['flipped'] = False
            st.session_state[state_key] = incoming
        database = st.session_state[state_key].copy()
        database['paleolatitude'] = paleolatitude(database['pole_lat'], database['pole_lon'], ref_lat, ref_lon)
        database['label'] = database['name'].astype(str) + ' [' + database['_record_id'].astype(str) + ']'
        st.success(f'Research database 已载入：{len(database)} 行')
    except Exception as exc:
        st.error(f'Research database 文件读取失败：{exc}')

if apwp.empty and database.empty:
    st.info('请先上传 APWP 数据、Research database，或两者都上传。')
    st.markdown('**字段要求**：APWP 至少需要 `longitude`, `latitude`, `age`；Research database 至少需要 `plat`/`pole_lat`, `plon`/`pole_lon`, `age`。支持 CSV 和 Excel。')
    st.stop()

# Filter ages based on available datasets.
ages = []
for frame in [apwp, database]:
    if not frame.empty and 'age' in frame:
        ages.extend(frame['age'].dropna().astype(float).tolist())
min_data_age, max_data_age = min(ages), max(ages)
with st.sidebar:
    st.subheader('年龄筛选')
    default_range = (float(min_data_age), float(max_data_age))
    if min_data_age == max_data_age:
        age_range = default_range
        st.caption(f'当前数据年龄：{min_data_age:g} Ma')
    else:
        age_range = st.slider('年龄范围 (Ma)', min_value=float(min_data_age), max_value=float(max_data_age), value=default_range, step=max((max_data_age-min_data_age)/200, 0.01))
    st.subheader('古纬度图 Y 轴')
    y_min = st.number_input('Y 轴最小值 (°)', value=-60.0, min_value=-90.0, max_value=90.0, step=5.0)
    y_max = st.number_input('Y 轴最大值 (°)', value=25.0, min_value=-90.0, max_value=90.0, step=5.0)

if y_min >= y_max:
    st.sidebar.warning('Y 轴最小值必须小于最大值。')
    y_min, y_max = -60.0, 25.0

apwp_f = apwp[apwp['age'].between(*age_range)].copy() if not apwp.empty else apwp.copy()
db_f = database[database['age'].between(*age_range)].copy() if not database.empty else database.copy()

# ----------------------------- Polarity reversal -----------------------------
if not database.empty:
    st.subheader('Research database：手动极性倒转')
    st.caption('勾选需要倒转的记录后点击按钮。倒转后的结果会保留在本次上传文件对应的会话状态中。')
    working = database.copy()
    if len(working):
        options = {f"{r['_record_id']} — {r['name']} — {r['age']:g} Ma (pole {r['pole_lat']:.1f}°, {r['pole_lon']:.1f}°)": str(r['_record_id']) for _, r in working.iterrows()}
        selected_labels = st.multiselect('选择要倒转的记录', list(options.keys()), key='flip_select')
        if st.button('倒转所选极点（对跖点）', key='flip_selected'):
            state_key = next((k for k in st.session_state.keys() if k.startswith('database_working_') and any(str(v) in k for v in [])), None)
            # Resolve state using uploaded file hash; this ensures updates persist on rerun.
            fp = hashlib.sha256(db_upload.getvalue()).hexdigest()[:16] if db_upload is not None else None
            state_key = f'database_working_{fp}' if fp else None
            if state_key and state_key in st.session_state:
                w = st.session_state[state_key].copy()
                ids = [options[s] for s in selected_labels]
                mask = w['_record_id'].isin(ids)
                lat_new, lon_new = antipode(w.loc[mask, 'pole_lat'].to_numpy(), w.loc[mask, 'pole_lon'].to_numpy())
                w.loc[mask, 'pole_lat'] = lat_new
                w.loc[mask, 'pole_lon'] = lon_new
                w.loc[mask, 'flipped'] = ~w.loc[mask, 'flipped'].astype(bool)
                st.session_state[state_key] = w
                st.rerun()
        if st.button('恢复数据库原始极点', key='reset_database'):
            fp = hashlib.sha256(db_upload.getvalue()).hexdigest()[:16] if db_upload is not None else None
            state_key = f'database_working_{fp}' if fp else None
            if state_key and state_key in st.session_state:
                original = standardize_database(read_table(db_upload))
                original['_record_id'] = [f'DB-{i+1:04d}' for i in range(len(original))]
                original['flipped'] = False
                st.session_state[state_key] = original
                st.rerun()

# Refresh database after polarity interaction state has been applied.
if db_upload is not None:
    fp = hashlib.sha256(db_upload.getvalue()).hexdigest()[:16]
    state_key = f'database_working_{fp}'
    if state_key in st.session_state:
        database = st.session_state[state_key].copy()
        database['paleolatitude'] = paleolatitude(database['pole_lat'], database['pole_lon'], ref_lat, ref_lon)
        database['label'] = database['name'].astype(str) + ' [' + database['_record_id'].astype(str) + ']'
        db_f = database[database['age'].between(*age_range)].copy()

# ----------------------------- Paleolatitude curve -----------------------------
st.header('1. 古纬度—年代曲线')
fig_curve = go.Figure()
if not apwp_f.empty:
    # Draw each uploaded APWP file as an independent curve, avoiding artificial
    # connecting lines between unrelated datasets.
    palette = ['#176b87', '#8b5e83', '#3a7d44', '#c06c45', '#5b6abf', '#8a7a36', '#397f83', '#b0526f']
    if 'source_file' not in apwp_f.columns:
        apwp_f['source_file'] = 'APWP'
    for curve_index, (source_name, source_curve) in enumerate(apwp_f.groupby('source_file', sort=False, dropna=False)):
        curve = source_curve.sort_values('age', ascending=False).copy()
        color = palette[curve_index % len(palette)]
        fill_color = f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.13)'
        if 'a95' in curve.columns and curve['a95'].notna().any():
            # Approximate paleolatitude uncertainty band; not rigorous spherical error propagation.
            err = curve['a95'].fillna(0).abs()
            fig_curve.add_trace(go.Scatter(x=curve['age'], y=curve['paleolatitude'] + err, mode='lines', line=dict(width=0, color=color), showlegend=False, hoverinfo='skip', name=f'{source_name} A95 upper'))
            fig_curve.add_trace(go.Scatter(x=curve['age'], y=curve['paleolatitude'] - err, mode='lines', line=dict(width=0, color=color), fill='tonexty', fillcolor=fill_color, showlegend=False, hoverinfo='skip', name=f'{source_name} approximate A95 band'))
        xerr = None
        lo, hi = get_age_bounds(curve)
        xerr_array = np.maximum(curve['age'].to_numpy() - lo.to_numpy(), hi.to_numpy() - curve['age'].to_numpy())
        if np.any(xerr_array > 0):
            xerr = dict(type='data', array=xerr_array, visible=True, thickness=1, width=2)
        yerr = None
        if 'a95' in curve.columns and curve['a95'].notna().any():
            yerr = dict(type='data', array=curve['a95'].fillna(0).abs().to_numpy(), visible=True, thickness=1, width=2)
        fig_curve.add_trace(go.Scatter(x=curve['age'], y=curve['paleolatitude'], mode='lines+markers', name=str(source_name), text=curve['label'], error_x=xerr, error_y=yerr, marker=dict(size=7, color=color, line=dict(color='white', width=1)), line=dict(width=2.6, color=color), hovertemplate='%{text}<br>Age: %{x:.2f} Ma<br>Paleolatitude: %{y:.2f}°<extra></extra>'))
if not db_f.empty:
    db_lo, db_hi = get_age_bounds(db_f)
    db_xerr_array = np.maximum(db_f['age'].to_numpy() - db_lo.to_numpy(), db_hi.to_numpy() - db_f['age'].to_numpy())
    db_xerr = dict(type='data', array=db_xerr_array, visible=True, thickness=1, width=2) if np.any(db_xerr_array > 0) else None
    db_yerr = None
    if 'a95' in db_f.columns and db_f['a95'].notna().any():
        db_yerr = dict(type='data', array=db_f['a95'].fillna(0).abs().to_numpy(), visible=True, thickness=1, width=2)
    fig_curve.add_trace(go.Scatter(x=db_f['age'], y=db_f['paleolatitude'], mode='markers', name='Research database', text=db_f['label'], error_x=db_xerr, error_y=db_yerr, marker=dict(size=10, symbol='diamond', color='#d66a52', line=dict(color='white', width=1.1)), customdata=np.stack([db_f['pole_lat'], db_f['pole_lon']], axis=-1), hovertemplate='%{text}<br>Age: %{x:.2f} Ma<br>Paleolatitude: %{y:.2f}°<br>Pole: %{customdata[0]:.2f}°, %{customdata[1]:.2f}°<extra></extra>'))
fig_curve.update_layout(title='Paleolatitude versus age', xaxis_title='Age (Ma)', yaxis_title='Paleolatitude (°)', hovermode='closest')
polish_figure(fig_curve, height=590)
fig_curve.update_xaxes(autorange='reversed', showgrid=True, gridcolor='#e7edf1', gridwidth=0.8, zeroline=False, ticks='outside', tickcolor='#8797a3', linecolor='#aab8c2', mirror=False)
fig_curve.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor='#e7edf1', gridwidth=0.8, zeroline=True, zerolinecolor='#8b9aa5', zerolinewidth=1.1, ticks='outside', tickcolor='#8797a3', linecolor='#aab8c2')
st.plotly_chart(fig_curve, use_container_width=True, config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "svg", "filename": "paleolatitude_age_curve", "scale": 2}})
vector_export(fig_curve, 'paleolatitude_age_curve', 'curve_export')

# ----------------------------- Stereonet -----------------------------
st.header('2. 古地磁极赤平投影图')
st.caption('采用 Schmidt 等面积投影。具有 A95 数据的极点显示相应的 95% 置信圆；为便于在同一网图中比较，南半球极点以其对跖点折叠到上半球。悬停可查看原始极点坐标和半球。红色圆点代表原始北半球极点，蓝色菱形代表原始南半球极点。')
point_frames = []
if not apwp_f.empty:
    apwp_cols = ['latitude', 'longitude', 'age', 'paleolatitude', 'label']
    if 'a95' in apwp_f.columns:
        apwp_cols.append('a95')
    t = apwp_f[apwp_cols].copy()
    if 'a95' not in t.columns:
        t['a95'] = np.nan
    t = t.rename(columns={'latitude':'pole_lat','longitude':'pole_lon'})
    t['source'] = 'APWP'
    point_frames.append(t)
if not db_f.empty:
    db_cols = ['pole_lat', 'pole_lon', 'age', 'paleolatitude', 'label']
    if 'a95' in db_f.columns:
        db_cols.append('a95')
    t = db_f[db_cols].copy()
    if 'a95' not in t.columns:
        t['a95'] = np.nan
    t['source'] = 'Research database'
    point_frames.append(t)
all_points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame(columns=['pole_lat','pole_lon','age','paleolatitude','label','source'])
if len(all_points):
    all_points['choice_label'] = [f"{i+1}. {row['source']} | {row['label']} | {row['age']:.2f} Ma | pole ({row['pole_lat']:.1f}°, {row['pole_lon']:.1f}°)" for i, row in all_points.iterrows()]
    fig_stereo = stereonet_figure(all_points.rename(columns={'choice_label':'label'}), ref_lat=ref_lat, ref_lon=ref_lon)
    st.plotly_chart(fig_stereo, use_container_width=True, config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "svg", "filename": "paleomagnetic_poles_stereonet", "scale": 2}})
    vector_export(fig_stereo, 'paleomagnetic_poles_schmidt_stereonet', 'stereo_export')
else:
    st.info('当前年龄范围内没有可绘制的极点。')

# ----------------------------- Small-circle comparison -----------------------------
st.header('3. 小圆弧分析：比较同一参考点下多个极点的古纬度')
st.markdown('对于给定参考点，具有相同古纬度的所有可能极点构成以参考点为球面中心的等角距小圆。可同时选择多个极点，并在同一张 Schmidt 等面积网投影图上显示各自对应的小圆。古纬度差按球面几何计算，不使用投影图上的平面距离代替。')
if len(all_points) >= 2:
    choices = all_points['choice_label'].tolist()
    default_choices = choices[:min(2, len(choices))]
    selected_choices = st.multiselect(
        '选择要同时比较的极点（可多选）', choices,
        default=default_choices, key='small_circle_multi_select'
    )
    if len(selected_choices) >= 2:
        selected_rows = all_points.loc[all_points['choice_label'].isin(selected_choices)].copy()
        # Preserve the order selected by the user rather than the original table order.
        selected_rows['_selection_order'] = selected_rows['choice_label'].map({v: i for i, v in enumerate(selected_choices)})
        selected_rows = selected_rows.sort_values('_selection_order').drop(columns='_selection_order')
        palette = ['#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#1f77b4', '#d62728']
        first_palat = None
        result_rows = []
        circle_rows = []
        for i, (_, row) in enumerate(selected_rows.iterrows()):
            palat = float(paleolatitude(row['pole_lat'], row['pole_lon'], ref_lat, ref_lon))
            dist = 90.0 - palat
            if first_palat is None:
                first_palat = palat
            color = palette[i % len(palette)]
            short_label = f'{i+1}. {row["source"]} | {row["label"]}'
            result_rows.append({
                '顺序': i + 1,
                '数据来源': row['source'],
                '极点': row['label'],
                '年龄 (Ma)': float(row['age']),
                '极点纬度 (°)': float(row['pole_lat']),
                '极点经度 (°)': float(row['pole_lon']),
                '古纬度 (°)': palat,
                '球面角距 (°)': dist,
                '相对第一个极点的古纬度差 (°)': palat - first_palat,
            })
            circle_rows.append((row, f'{i+1}. small circle ({dist:.1f}°)', color))
        st.dataframe(pd.DataFrame(result_rows).style.format({
            '年龄 (Ma)': '{:.2f}', '极点纬度 (°)': '{:.2f}', '极点经度 (°)': '{:.2f}',
            '古纬度 (°)': '{:.2f}', '球面角距 (°)': '{:.2f}',
            '相对第一个极点的古纬度差 (°)': '{:+.2f}'
        }), use_container_width=True, hide_index=True)
        st.caption('“相对第一个极点的古纬度差”以你在上方选择列表中排在第一位的极点为基准。')
        if st.checkbox('在同一张赤平投影图上叠加所选极点及其等古纬度小圆', value=True, key='show_small_circles_multi'):
            fig_small = stereonet_figure(
                selected_rows, ref_lat=ref_lat, ref_lon=ref_lon,
                circle_rows=circle_rows
            )
            fig_small.update_layout(title='Selected poles and equal-paleolatitude small circles on Schmidt net')
            polish_figure(fig_small, height=700)
            st.plotly_chart(fig_small, use_container_width=True, config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "svg", "filename": "paleomagnetic_plot", "scale": 2}})
            vector_export(fig_small, 'small_circle_paleolatitude_comparison', 'smallcircle_export')
    elif len(selected_choices) == 1:
        st.info('请至少再选择一个极点，才能比较不同极点的古纬度。')
    else:
        st.info('请至少选择两个极点进行比较。')
else:
    st.info('至少需要两个极点才能进行小圆弧比较。')

# ----------------------------- Data tables and downloads -----------------------------
st.header('4. 数据表与导出')
if not apwp.empty:
    with st.expander('查看 APWP 数据（含计算古纬度）'):
        st.dataframe(apwp, use_container_width=True)
    st.download_button('下载 APWP 计算结果 CSV', data=apwp.to_csv(index=False).encode('utf-8-sig'), file_name='apwp_paleolatitude_results.csv', mime='text/csv', key='apwp_csv')
if not database.empty:
    with st.expander('查看 Research database（含倒转状态与计算古纬度）'):
        st.dataframe(database, use_container_width=True)
    st.download_button('下载 Research database 计算结果 CSV', data=database.to_csv(index=False).encode('utf-8-sig'), file_name='research_database_paleolatitude_results.csv', mime='text/csv', key='database_csv')
if len(all_points):
    st.download_button('下载当前筛选的全部极点 CSV', data=all_points.to_csv(index=False).encode('utf-8-sig'), file_name='filtered_paleomagnetic_poles.csv', mime='text/csv', key='all_points_csv')

with st.expander('计算方法与注意事项'):
    st.markdown('''
- 古纬度按 `paleolatitude = 90° − angular_distance(reference point, paleomagnetic pole)` 计算。
- 极点倒转采用对跖变换：纬度变号，经度加 180° 并归一化。
- 赤平投影使用 Schmidt 等面积网；南半球极点为便于同图显示折叠到上半球。原始坐标保留在悬停信息中。
- A95 用于赤平投影图上的极点置信圆，并用于古纬度误差棒和阴影带的近似表示；后两者**不是严格的球面误差传播**。若需要严谨的古纬度不确定度，应基于极点置信锥在球面上进行蒙特卡洛传播。
- 年龄误差只有在表格提供 `min_age`/`max_age` 或 `age_error` 时才会绘制。
- SVG/PDF 导出需要安装与当前 Plotly 兼容的 `kaleido`。导出失败时，页面会给出错误信息文件和安装提示。
''')
