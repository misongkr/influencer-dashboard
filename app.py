import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import subprocess

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="인플루언서 전략 대시보드",
    layout="wide"
)

def check_password():
    if "authed" not in st.session_state:
        st.session_state.authed = False

    if not st.session_state.authed:
        st.sidebar.markdown("### 🔒 Access")
        pwd = st.sidebar.text_input("비밀번호", type="password")

        if pwd == st.secrets["APP_PASSWORD"]:
            st.session_state.authed = True
            st.rerun()
        else:
            st.sidebar.caption("비밀번호를 입력해주세요")
            st.stop()

check_password()


PLATFORM_COLORS = {
    "치지직": "#A8E05F",  # 연두
    "SOOP": "#8FD3F4",   # 하늘
}

# 기여 타입 컬러(플랫폼 컬러와 겹치지 않게)
TYPE_BADGE = {
    "시간형": {"bg": "#C4B5FD", "fg": "#312E81"},   # 퍼플
    "파워형": {"bg": "#FCA5A5", "fg": "#7F1D1D"},   # 레드
    "밸런스형": {"bg": "#D1D5DB", "fg": "#111827"}, # 그레이
}

# 파일명에서 YYYYMMDD 추출
DATE_RE = re.compile(r"(20\d{6})")



# -----------------------------
# Utils
# -----------------------------
def parse_snapshot_date_from_name(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def parse_period_from_filename(name: str):
    """
    반환:
    - ("weekly", date)   → YYYYMMDD
    - ("monthly", date)  → YYYYMM (해당 월 1일로 normalize)
    - (None, None)
    """
    stem = Path(name).stem

    # 1) 주간: YYYYMMDD (8자리) 먼저
    m = re.search(r"(20\d{6}\d{2})", stem)  # YYYYMMDD
    if m:
        token = m.group(1)
        try:
            return "weekly", datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            pass

    # 2) 월간: YYYYMM (6자리)
    m = re.search(r"(20\d{4}\d{2})(?!\d)", stem)  # YYYYMM (뒤에 숫자 더 붙으면 제외)
    if m:
        token = m.group(1)
        try:
            return "monthly", datetime.strptime(token, "%Y%m").date()
        except ValueError:
            pass

    return None, None



def to_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in ("nan", "none", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def pct_change(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    return (curr - prev) / prev * 100.0


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.0f}%"

def kpi_with_wow(curr, prev, decimals=0):
    """
    decimals=0: 정수 반올림 표시
    decimals=1: 소수점 1자리 표시(가중 평균 시청자용)
    반환 예) 25,664 (<span ...>(↑6%)</span>)
    """
    if curr is None or pd.isna(curr):
        return "N/A"

    # 값 포맷
    if decimals == 0:
        curr_txt = f"{float(curr):,.0f}"
    else:
        curr_txt = f"{float(curr):,.{decimals}f}"

    # 전주 값 없으면
    if prev is None or pd.isna(prev) or prev == 0:
        return f"{curr_txt} (N/A)"

    wow = pct_change(curr, prev)
    if wow is None or pd.isna(wow):
        return f"{curr_txt} (N/A)"

    if wow > 0:
        wow_txt = f"<span style='color:#DC2626;'>(↑{wow:.0f}%)</span>"
    elif wow < 0:
        wow_txt = f"<span style='color:#2563EB;'>(↓{abs(wow):.0f}%)</span>"
    else:
        wow_txt = "(0%)"

    return f"{curr_txt} {wow_txt}"



def ensure_columns(df, required, df_name="df"):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{df_name}] missing columns: {missing}")


def to_monday(dt_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt_series, errors="coerce")
    # Monday=0
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize()


# -----------------------------
# Loaders (운영 룰 반영)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_latest_prev_streamer_softcon_and_cat(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    def read_csv(fp: Path):
        df = pd.read_csv(fp, encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # -------------------------------------------------
    # filename 파서: YYYYMMDD_HHMMSS / YYYYMMDDHHMMSS / YYYYMMDD
    # -------------------------------------------------
    def parse_dt_from_stem(stem: str):
        """
        우선순위:
        1) YYYYMMDD[_-]HHMMSS  (예: 20260216_101535)
        2) YYYYMMDDHHMMSS      (예: 20260216101535)
        3) YYYYMMDD            (예: 20260216)
        """
        s = stem

        m = re.search(r"(20\d{6})[_-](\d{6})", s)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass

        m = re.search(r"(20\d{6})(\d{6})", s)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass

        m = re.search(r"(20\d{6})", s)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d")
            except ValueError:
                pass

        return None

    def pick_latest_file(pattern: str):
        files = list(data_path.glob(pattern))
        if not files:
            return None

        pairs = []
        for f in files:
            dt = parse_dt_from_stem(f.stem)
            if dt is not None:
                pairs.append((dt, f))

        # 파일명에서 날짜 파싱 안 되면 수정시간(mtime)으로 fallback
        if not pairs:
            return max(files, key=lambda p: p.stat().st_mtime)

        pairs.sort(key=lambda x: x[0])
        return pairs[-1][1]

    def pick_latest_prev_files(pattern: str):
        """
        pattern에 해당하는 파일들 중 최신/전주(바로 이전) 파일을 반환
        return: (latest_dt, latest_fp, prev_dt, prev_fp)
        """
        files = list(data_path.glob(pattern))
        if not files:
            return (None, None, None, None)

        pairs = []
        for f in files:
            dt = parse_dt_from_stem(f.stem)
            if dt is not None:
                pairs.append((dt, f))

        if not pairs:
            # dt 파싱 실패 -> mtime 기반
            files = sorted(files, key=lambda p: p.stat().st_mtime)
            latest_fp = files[-1]
            prev_fp = files[-2] if len(files) >= 2 else None
            return (None, latest_fp, None, prev_fp)

        pairs.sort(key=lambda x: x[0])
        latest_dt, latest_fp = pairs[-1]
        prev_dt, prev_fp = (pairs[-2] if len(pairs) >= 2 else (None, None))
        return (latest_dt, latest_fp, prev_dt, prev_fp)

    # -------------------------------------------------
    # (선택) 월간/주간 구분: 기존 로직 유지
    # 월간은 YYYYMM만 있는 파일도 있으니 기존 parse_period_from_filename 사용
    # -------------------------------------------------
    def period_files(pattern: str, kind: str):
        files = list(data_path.glob(pattern))
        pairs = []
        for f in files:
            k, d = parse_period_from_filename(f.name)
            if k == kind and d is not None:
                # 월간은 date만 있으니 (date, file)로 정렬
                pairs.append((d, f))
        pairs.sort(key=lambda x: x[0])
        return pairs

    # ✅ 주간 스트리머 (날짜+시간까지 고려)
    sr_latest_dt, sr_latest_fp, sr_prev_dt, sr_prev_fp = pick_latest_prev_files("스트리머_랭킹*.csv")

    # ✅ 월간 스트리머 (파일명이 YYYYMM만 있는 경우)
    sr_m_pairs = period_files("스트리머_랭킹*.csv", "monthly")
    sr_m_latest_date, sr_m_latest_fp = (sr_m_pairs[-1] if len(sr_m_pairs) >= 1 else (None, None))
    sr_m_prev_date, sr_m_prev_fp = (sr_m_pairs[-2] if len(sr_m_pairs) >= 2 else (None, None))

    # ✅ 소프트콘(주간) (날짜+시간까지 고려)
    sc_latest_dt, sc_latest_fp, sc_prev_dt, sc_prev_fp = pick_latest_prev_files("소프트콘_랭킹*.csv")

    # ✅ 카테고리 통계: 가장 최근 생성본(파일명 날짜+시간) 선택 (핵심 수정)
    cat_fp = pick_latest_file("카테고리_플랫폼별_통계*.csv")

    # 유효성 체크
    if (sr_latest_fp is None) or (cat_fp is None):
        return None

    # streamer_latest_date / prev_date는 기존 타입(date) 기대하니까 date로 normalize
    streamer_latest_date = sr_latest_dt.date() if isinstance(sr_latest_dt, datetime) else None
    streamer_prev_date = sr_prev_dt.date() if isinstance(sr_prev_dt, datetime) else None

    softcon_latest_date = sc_latest_dt.date() if isinstance(sc_latest_dt, datetime) else None
    softcon_prev_date = sc_prev_dt.date() if isinstance(sc_prev_dt, datetime) else None

    return {
        "streamer_latest_date": streamer_latest_date,
        "streamer_prev_date": streamer_prev_date,
        "streamer_latest": read_csv(sr_latest_fp),
        "streamer_prev": read_csv(sr_prev_fp) if sr_prev_fp else None,

        "streamer_monthly_latest_date": sr_m_latest_date,
        "streamer_monthly_prev_date": sr_m_prev_date,
        "streamer_monthly_latest": read_csv(sr_m_latest_fp) if sr_m_latest_fp else None,
        "streamer_monthly_prev": read_csv(sr_m_prev_fp) if sr_m_prev_fp else None,

        "softcon_latest_date": softcon_latest_date,
        "softcon_prev_date": softcon_prev_date,
        "softcon_latest": read_csv(sc_latest_fp) if sc_latest_fp else None,
        "softcon_prev": read_csv(sc_prev_fp) if sc_prev_fp else None,

        "cat_current": read_csv(cat_fp),
        "cat_filename": cat_fp.name,
    }


def split_curr_prev_weeks_from_cat_stats(cat_stats_raw: pd.DataFrame):
    """
    카테고리_플랫폼별_통계(누적 1파일)에서
    최신 주(월요일) / 전주 주(월요일) 데이터프레임을 반환
    """
    cs = normalize_category_platform_stats(cat_stats_raw)
    cs = cs.dropna(subset=["날짜"])
    cs["주차(월)"] = to_monday(cs["날짜"])

    weeks = sorted(cs["주차(월)"].dropna().unique())
    curr_week = weeks[-1] if len(weeks) >= 1 else None
    prev_week = weeks[-2] if len(weeks) >= 2 else None

    cs_curr = cs[cs["주차(월)"] == curr_week].copy() if curr_week is not None else cs.iloc[0:0].copy()
    cs_prev = cs[cs["주차(월)"] == prev_week].copy() if prev_week is not None else cs.iloc[0:0].copy()
    return curr_week, prev_week, cs_curr, cs_prev


# -----------------------------
# Normalizers
# -----------------------------
def normalize_streamer_rank(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "최고 시청자": "최고시청자",
        "평균 시청자": "평균시청자",
        "방송 시간": "방송시간",
    }
    df = df.rename(columns=rename_map).copy()

    required = ["플랫폼", "스트리머", "방송시간", "평균시청자", "뷰어십"]
    ensure_columns(df, required, "streamer_rank")

    for c in ["방송시간", "최고시청자", "평균시청자", "뷰어십"]:
        if c in df.columns:
            df[c] = df[c].map(to_number)

    df["플랫폼"] = df["플랫폼"].astype(str).str.strip()
    df["스트리머"] = df["스트리머"].astype(str).str.strip()
    df = df.dropna(subset=["플랫폼", "스트리머"])
    return df


def normalize_category_platform_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_candidates = {
        "platform": ["플랫폼", "platform"],
        "date": ["날짜", "일자", "date"],
        "stream_time": ["방송시간", "방송 시간"],
        "avg_viewers": ["시청자수 평균", "평균 시청자", "평균시청자"],
        "avg_chat": ["6분당 채팅수 평균", "평균 채팅수", "6분 평균"],
        "viewership": ["뷰어십", "누적 뷰어십"],
    }

    def pick(col_list):
        for c in col_list:
            if c in df.columns:
                return c
        return None

    platform_col = pick(col_candidates["platform"])
    date_col = pick(col_candidates["date"])
    stream_time_col = pick(col_candidates["stream_time"])
    avg_viewers_col = pick(col_candidates["avg_viewers"])
    avg_chat_col = pick(col_candidates["avg_chat"])
    viewership_col = pick(col_candidates["viewership"])

    out = pd.DataFrame()

    # platform은 필수
    if platform_col is None:
        out["플랫폼"] = np.nan
    else:
        out["플랫폼"] = df[platform_col].astype(str).str.strip()

    # -------------------------
    # ✅ 날짜 파싱 (핵심)
    # -------------------------
    if date_col is None:
        out["날짜"] = pd.NaT
    else:
        raw = df[date_col]

        # 1) 숫자형(엑셀 시리얼/yyyymmdd) 가능성
        if pd.api.types.is_numeric_dtype(raw):
            s = raw.copy()

            # excel serial로 보이는 값 (대략 30000~80000)
            mask_excel = s.between(30000, 80000, inclusive="both")
            out_date = pd.Series([pd.NaT] * len(s), index=df.index)

            if mask_excel.any():
                out_date.loc[mask_excel] = pd.to_datetime(
                    s.loc[mask_excel],
                    unit="D",
                    origin="1899-12-30",
                    errors="coerce"
                )

            # 나머지는 yyyymmdd 시도
            rest = ~mask_excel
            if rest.any():
                out_date.loc[rest] = pd.to_datetime(
                    s.loc[rest].astype("Int64").astype(str),
                    format="%Y%m%d",
                    errors="coerce"
                )

            out["날짜"] = out_date

        else:
            raw_date = raw.astype(str).str.strip()

            # 괄호/요일 제거
            raw_date = raw_date.str.replace(r"\s*\([^)]*\)\s*", "", regex=True)

            # 한글/기타 텍스트 제거
            raw_date = raw_date.str.replace(r"[가-힣]+", "", regex=True).str.strip()

            # 구분자 통일
            raw_date = raw_date.str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)

            # 공백 있으면 앞 토큰만
            raw_date2 = raw_date.str.split().str[0]

            # 1차: YYYY-MM-DD
            out_dt = pd.to_datetime(raw_date2, format="%Y-%m-%d", errors="coerce")

            # 2차: YYYYMMDD
            mask = out_dt.isna()
            if mask.any():
                out_dt.loc[mask] = pd.to_datetime(raw_date2.loc[mask], format="%Y%m%d", errors="coerce")

            out["날짜"] = out_dt

    # -------------------------
    # 나머지 수치 컬럼
    # -------------------------
    out["방송시간"] = df[stream_time_col].map(to_number) if stream_time_col else np.nan
    out["평균시청자"] = df[avg_viewers_col].map(to_number) if avg_viewers_col else np.nan
    out["평균채팅"] = df[avg_chat_col].map(to_number) if avg_chat_col else np.nan
    out["뷰어십"] = df[viewership_col].map(to_number) if viewership_col else np.nan

    out = out.dropna(subset=["플랫폼"])
    return out


# -----------------------------
# Metrics
# -----------------------------
def compute_overview_metrics(streamer_rank: pd.DataFrame, cat_stats_week: pd.DataFrame):
    """
    cat_stats_week: 반드시 '해당 주차' 데이터만 들어오도록 (메인에서 필터링)
    """
    sr = normalize_streamer_rank(streamer_rank)

    cs = cat_stats_week.copy()
    # cat_stats_week는 split 함수에서 이미 normalize_category_platform_stats() 결과여야 하지만,
    # 혹시 raw가 들어오면 대비
    if ("평균시청자" not in cs.columns) or ("방송시간" not in cs.columns) or ("플랫폼" not in cs.columns):
        cs = normalize_category_platform_stats(cs)

    # --- Totals (스트리머 랭킹 기반) ---
    total_stream_time = sr["방송시간"].sum(skipna=True)
    total_viewership = sr["뷰어십"].sum(skipna=True)

    # 가중 평균 시청자(뷰어십/방송시간) = 지금까지 '평균 시청자=8'의 정체
    weighted_avg_viewers = (total_viewership / total_stream_time) if total_stream_time and total_stream_time > 0 else np.nan

    # --- 평균 시청자(카테고리 통계 기반, ACV) ---
    # 두 플랫폼을 "합산"하는 개념으로: 방송시간으로 가중평균
    cs_v = cs.dropna(subset=["평균시청자", "방송시간"]).copy()
    cs_v = cs_v[(cs_v["평균시청자"] > 0) & (cs_v["방송시간"] > 0)]

    if len(cs_v) > 0:
        cat_avg_viewers = (cs_v["평균시청자"] * cs_v["방송시간"]).sum() / cs_v["방송시간"].sum()
    else:
        cat_avg_viewers = np.nan

    # --- Platform comparison (스트리머 랭킹 기반) ---
    platform_df = (
        sr.groupby("플랫폼", as_index=False)
          .agg(방송시간=("방송시간", "sum"), 뷰어십=("뷰어십", "sum"))
    )
    platform_df["평균시청자"] = platform_df.apply(
        lambda r: (r["뷰어십"] / r["방송시간"]) if r["방송시간"] and r["방송시간"] > 0 else np.nan,
        axis=1
    )

    # --- Top streamers (스트리머 랭킹 기반) ---
    top_df = (
        sr.groupby(["플랫폼", "스트리머"], as_index=False)
          .agg(
              방송시간=("방송시간", "sum"),
              평균시청자=("평균시청자", "mean"),
              뷰어십=("뷰어십", "sum"),
          )
    )

    top_df["기여율"] = top_df["뷰어십"] / total_viewership if total_viewership and total_viewership > 0 else np.nan

    # 기여 타입 분류(스냅샷 내 상대 랭크)
    top_df["방송시간_rank"] = top_df["방송시간"].rank(pct=True, ascending=False)
    top_df["평균시청자_rank"] = top_df["평균시청자"].rank(pct=True, ascending=False)

    def classify(row):
        # 상위 기준: top 25% (<=0.25), 하위 기준: bottom 50% (>0.5)
        if row["방송시간_rank"] <= 0.25 and row["평균시청자_rank"] > 0.5:
            return "시간형"
        elif row["방송시간_rank"] > 0.5 and row["평균시청자_rank"] <= 0.25:
            return "파워형"
        else:
            return "밸런스형"

    top_df["기여 타입"] = top_df.apply(classify, axis=1)

    # Top10: 뷰어십 기준 내림차순
    top10 = top_df.sort_values("뷰어십", ascending=False).head(10).copy()

    extra = {
        "top_streamer_name": top10.iloc[0]["스트리머"] if len(top10) else None,
        "top_streamer_share": top10.iloc[0]["기여율"] if len(top10) else None,
    }

    # --- KPI (표시용) ---
    kpis = {
        "총 뷰어십": total_viewership,
        "평균 시청자(카테고리)": cat_avg_viewers,                 # 사람들이 기대하는 평균시청자(ACV)
        "총 방송시간": total_stream_time,
        "가중 평균 시청자(뷰어십/시간)": weighted_avg_viewers,     # 기존 8.x는 여기로
    }

    return kpis, platform_df, top10, top_df, extra



def platform_wow_increment(curr_pf: pd.DataFrame, prev_pf: pd.DataFrame) -> pd.DataFrame:
    a = curr_pf[["플랫폼", "뷰어십"]].rename(columns={"뷰어십": "curr_viewership"})
    b = prev_pf[["플랫폼", "뷰어십"]].rename(columns={"뷰어십": "prev_viewership"})
    m = a.merge(b, on="플랫폼", how="outer").fillna(0)

    m["delta_viewership"] = m["curr_viewership"] - m["prev_viewership"]
    total_delta = m["delta_viewership"].sum()
    m["delta_share"] = np.where(total_delta != 0, m["delta_viewership"] / total_delta, np.nan)
    m["wow_pct"] = np.where(
        m["prev_viewership"] != 0,
        (m["curr_viewership"] - m["prev_viewership"]) / m["prev_viewership"] * 100,
        np.nan
    )
    return m.sort_values("delta_viewership", ascending=False).reset_index(drop=True)


def build_summary_text(curr, prev):
    """
    리더용 인사이트형 요약 3~4줄:
    1) 전체 흐름(What) + 주도 플랫폼(Where)
    2) 플랫폼 효율/볼륨 관점(Where, Why)
    3) Top 스트리머 구조(Who)
    4) 소비 형태 해석(So what) - 카테고리 평균 vs 가중 평균
    """
    lines = []

    # -------------------------
    # 0) 안전 장치
    # -------------------------
    curr_k = curr.get("kpis", {})
    curr_pf = curr.get("platform_df", pd.DataFrame()).copy()
    curr_ex = curr.get("extra", {})

    prev_k = prev.get("kpis", {}) if prev else {}
    prev_pf = prev.get("platform_df", pd.DataFrame()).copy() if prev else None

    # -------------------------
    # 1) 전체 뷰어십 WoW + 증가/감소 주도 플랫폼
    # -------------------------
    curr_v = curr_k.get("총 뷰어십", np.nan)
    prev_v = prev_k.get("총 뷰어십", np.nan) if prev else None
    wow_v = pct_change(curr_v, prev_v) if prev else None

    if wow_v is None:
        lines.append("• 전체 뷰어십: 전주 데이터가 없어 WoW 계산 불가 (N/A)")
    else:
        # 증가/감소 기여 플랫폼
        inc = platform_wow_increment(curr_pf, prev_pf) if prev_pf is not None else None
        if inc is not None and len(inc) > 0:
            lead_platform = inc.iloc[0]["플랫폼"]
            lead_share = inc.iloc[0]["delta_share"]
            # 문장 톤
            if wow_v < 0:
                tone = "감소했으며"
            elif wow_v > 0:
                tone = "증가했으며"
            else:
                tone = "변화가 거의 없었으며"

            if lead_platform and not pd.isna(lead_share):
                lines.append(
                    f"• 전체 뷰어십은 전주 대비 {fmt_pct(wow_v)}로 {tone} "
                    f"{lead_platform}이(가) 변화분의 {lead_share*100:.0f}%를 주도함."
                )
            else:
                lines.append(f"• 전체 뷰어십은 전주 대비 {fmt_pct(wow_v)}.")
        else:
            lines.append(f"• 전체 뷰어십은 전주 대비 {fmt_pct(wow_v)}.")

    # -------------------------
    # 2) 플랫폼 구조(볼륨 vs 효율) 해석 한 줄
    # - 뷰어십(볼륨) 1위 플랫폼
    # - 평균시청자(효율) 1위 플랫폼
    # -------------------------
    if len(curr_pf) >= 1 and ("뷰어십" in curr_pf.columns) and ("평균시청자" in curr_pf.columns):
        pf_sorted_v = curr_pf.sort_values("뷰어십", ascending=False)
        pf_sorted_eff = curr_pf.sort_values("평균시청자", ascending=False)

        vol_leader = pf_sorted_v.iloc[0]["플랫폼"]
        eff_leader = pf_sorted_eff.iloc[0]["플랫폼"]

        if vol_leader == eff_leader:
            lines.append(f"• 플랫폼 구조: {vol_leader}이(가) 볼륨(뷰어십)과 효율(평균 시청자) 모두 우세한 구간.")
        else:
            lines.append(
                f"• 플랫폼 구조: 볼륨(뷰어십)은 {vol_leader} 중심, "
                f"효율(평균 시청자)은 {eff_leader} 우세 → 유입 vs 효율이 분리된 구조."
            )

    # -------------------------
    # 3) Top 스트리머 구조(집중도/의존도) 해석
    # -------------------------
    ts = curr_ex.get("top_streamer_name")
    ts_share = curr_ex.get("top_streamer_share")

    if ts and ts_share is not None and not pd.isna(ts_share):
        # 의존도 해석(임계값은 운영하면서 조정 가능)
        if ts_share >= 0.15:
            dep = "상위 스트리머 의존도가 높은 편"
        elif ts_share >= 0.10:
            dep = "상위 스트리머 중심 구조가 유지"
        else:
            dep = "상위 스트리머 집중도는 과도하지 않음"

        lines.append(f"• Top 스트리머 {ts}가 전체 뷰어십의 {ts_share*100:.0f}%를 기여 → {dep}.")

    # -------------------------
    # 4) 소비 형태 해석(카테고리 평균 vs 가중 평균)
    # -------------------------
    cat_avg = curr_k.get("평균 시청자(카테고리)", np.nan)
    w_avg = curr_k.get("가중 평균 시청자(뷰어십/시간)", np.nan)

    if (not pd.isna(cat_avg)) and (not pd.isna(w_avg)) and cat_avg > 0:
        ratio = w_avg / cat_avg  # 작을수록 '관전형/분산' 해석
        # 경험적으로 임계값 조정 추천. 우선 합리적 디폴트:
        if ratio < 0.20:
            insight = "카테고리 평균 대비 가중 평균 격차가 커 관전형 소비(분산 시청) 비중이 높은 주간으로 해석."
        elif ratio < 0.35:
            insight = "카테고리 평균 대비 가중 평균 격차가 존재 → 분산 시청 성격이 우세하나 핵심 구간도 유지."
        else:
            insight = "가중 평균이 카테고리 평균에 근접 → 특정 구간 집중/충성 시청 성격이 강화된 흐름."

        lines.append(f"• 소비 형태: {insight}")

    return lines

# -----------------------------
# Recommendation (캠페인 목적 -> 후보풀)
# -----------------------------
PURPOSES = {
    "대형 업데이트/쇼케이스(파급력 우선)": {
        "weights": {"power": 0.55, "eff": 0.25, "stability": 0.20},
        "desc": "최대/평균 시청자와 총 기여(뷰어십) 중심. 큰 무대에서 확실한 카드."
    },
    "효율 중심(예산/시간 제한)": {
        "weights": {"power": 0.25, "eff": 0.55, "stability": 0.20},
        "desc": "방송시간 대비 성과(뷰어십/시간) 중심. 비용/시간 대비 효율 최적화."
    },
    "신규/중견 발굴(성장/기회)": {
        "weights": {"power": 0.25, "eff": 0.35, "stability": 0.40},
        "desc": "전주 대비 개선/안정(추세) 비중을 높여, 다음 달 성장 후보를 찾음."
    },
    "리스크 낮은 운영형(안정/지속)": {
        "weights": {"power": 0.25, "eff": 0.25, "stability": 0.50},
        "desc": "전주 대비 하락 리스크가 낮고 성과가 유지되는 스트리머 중심."
    },
}

def compute_streamer_features(sr_df: pd.DataFrame) -> pd.DataFrame:
    """주간(스냅샷) 스트리머 단위 피처 생성"""
    sr = normalize_streamer_rank(sr_df)

    g = (
        sr.groupby(["플랫폼", "스트리머"], as_index=False)
          .agg(
              방송시간=("방송시간", "sum"),
              평균시청자=("평균시청자", "mean"),
              뷰어십=("뷰어십", "sum"),
              최고시청자=("최고시청자", "max") if "최고시청자" in sr.columns else ("평균시청자", "max"),
          )
    )

    total_viewership = g["뷰어십"].sum() if g["뷰어십"].notna().any() else 0.0
    g["기여율"] = np.where(total_viewership > 0, g["뷰어십"] / total_viewership, np.nan)

    # 효율(시간 대비 성과)
    g["효율(뷰어십/시간)"] = np.where(g["방송시간"] > 0, g["뷰어십"] / g["방송시간"], np.nan)

    # 피크 의존도(이벤트형/스파이크 성향 탐지용)
    g["피크비율(최고/평균)"] = np.where(
        (g["평균시청자"] > 0) & g["최고시청자"].notna(),
        g["최고시청자"] / g["평균시청자"],
        np.nan
    )

    # 현재 너 코드의 분류 로직 재사용(상대 랭크 기반)
    g["방송시간_rank"] = g["방송시간"].rank(pct=True, ascending=False)
    g["평균시청자_rank"] = g["평균시청자"].rank(pct=True, ascending=False)

    def classify(row):
        if row["방송시간_rank"] <= 0.25 and row["평균시청자_rank"] > 0.5:
            return "시간형"
        elif row["방송시간_rank"] > 0.5 and row["평균시청자_rank"] <= 0.25:
            return "파워형"
        else:
            return "밸런스형"

    g["기여 타입"] = g.apply(classify, axis=1)

    return g


def attach_prev_wow(curr_feat: pd.DataFrame, prev_feat: pd.DataFrame | None) -> pd.DataFrame:
    """전주 피처가 있으면 WoW(뷰어십/평균시청자) 계산해서 안정성 시그널로 사용"""
    out = curr_feat.copy()
    if prev_feat is None or len(prev_feat) == 0:
        out["뷰어십_WoW(%)"] = np.nan
        out["평균시청자_WoW(%)"] = np.nan
        return out

    a = out.rename(columns={"뷰어십": "curr_viewership", "평균시청자": "curr_avg"})
    b = prev_feat[["플랫폼", "스트리머", "뷰어십", "평균시청자"]].rename(
        columns={"뷰어십": "prev_viewership", "평균시청자": "prev_avg"}
    )

    m = a.merge(b, on=["플랫폼", "스트리머"], how="left")
    m["뷰어십_WoW(%)"] = np.where(
        (m["prev_viewership"].notna()) & (m["prev_viewership"] > 0),
        (m["curr_viewership"] - m["prev_viewership"]) / m["prev_viewership"] * 100,
        np.nan
    )
    m["평균시청자_WoW(%)"] = np.where(
        (m["prev_avg"].notna()) & (m["prev_avg"] > 0),
        (m["curr_avg"] - m["prev_avg"]) / m["prev_avg"] * 100,
        np.nan
    )

    # 원래 컬럼명 복구
    m = m.rename(columns={"curr_viewership": "뷰어십", "curr_avg": "평균시청자"})
    return m


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def recommend_streamers(curr_feat: pd.DataFrame, purpose_key: str, top_n: int = 20) -> pd.DataFrame:
    """캠페인 목적별 추천 스코어링 + 추천 사유 생성"""
    cfg = PURPOSES[purpose_key]
    w = cfg["weights"]

    df = curr_feat.copy()

    # --- score components ---
    # power: 평균시청자 + 기여율 + (옵션) 최고시청자
    df["_power"] = (
        0.55 * zscore(df["평균시청자"]) +
        0.30 * zscore(df["기여율"].fillna(0)) +
        0.15 * zscore(df["최고시청자"].fillna(df["평균시청자"]))
    )

    # efficiency: 뷰어십/시간
    df["_eff"] = zscore(df["효율(뷰어십/시간)"])

    # stability: 전주 대비 변화가 극단적이지 않고(변동 낮음), 유지/상승이면 가점
    # 전주 데이터 없으면 0으로 두고, 목적 가중치가 있어도 영향이 크지 않게 설계
    wow = df["뷰어십_WoW(%)"].copy()
    df["_stability"] = np.where(wow.isna(), 0.0, -np.abs(zscore(wow)) + 0.3 * zscore(wow))

    df["추천점수"] = (
        w["power"] * df["_power"] +
        w["eff"] * df["_eff"] +
        w["stability"] * df["_stability"]
    )

    # --- explanation (추천 사유) ---
    # 퍼센타일 기반으로 '사유 문장'을 만들면 UX에서 설득력이 좋아짐
    df["eff_pct"] = df["효율(뷰어십/시간)"].rank(pct=True) * 100
    df["power_pct"] = df["평균시청자"].rank(pct=True) * 100
    df["share_pct"] = df["기여율"].rank(pct=True) * 100

    def build_reason(r):
        reasons = []
        # 목적별로 보여주는 포인트를 다르게
        if w["power"] >= w["eff"]:
            reasons.append(f"평균 시청자 상위 {100 - int(r['power_pct']):d}%")
            reasons.append(f"기여율 상위 {100 - int(r['share_pct']):d}%")
        else:
            reasons.append(f"효율(뷰어십/시간) 상위 {100 - int(r['eff_pct']):d}%")
            reasons.append(f"방송시간 {r['방송시간']:.1f}h 대비 뷰어십 {r['뷰어십']:,.0f}")

        # 피크 성향(이벤트형) 보조 설명
        if pd.notna(r["피크비율(최고/평균)"]) and r["피크비율(최고/평균)"] >= 2.0:
            reasons.append("피크형(이벤트/스파이크 성향)")

        # 전주 정보가 있으면 안정성/추세 문장
        if pd.notna(r["뷰어십_WoW(%)"]):
            wow_txt = f"{r['뷰어십_WoW(%)']:+.0f}%"
            if r["뷰어십_WoW(%)"] >= 10:
                reasons.append(f"전주 대비 뷰어십 상승({wow_txt})")
            elif r["뷰어십_WoW(%)"] <= -10:
                reasons.append(f"전주 대비 뷰어십 하락({wow_txt})")
            else:
                reasons.append(f"전주 대비 뷰어십 안정({wow_txt})")

        # 타입 배지 의미
        reasons.append(f"기여 타입: {r['기여 타입']}")
        return " · ".join(reasons[:4])

    df["추천 사유"] = df.apply(build_reason, axis=1)

    # 정렬 & 출력 컬럼
    out = df.sort_values("추천점수", ascending=False).head(top_n).copy()
    return out[
        ["플랫폼", "스트리머", "추천점수", "뷰어십", "기여율", "방송시간", "평균시청자", "효율(뷰어십/시간)", "기여 타입", "추천 사유"]
    ]


# -----------------------------
# UI helpers
# -----------------------------
def bar_chart(df, x, y, title):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{x}:N",
                sort="-y",
                title="플랫폼",
                axis=alt.Axis(labelAngle=0, labelPadding=8, titlePadding=12, labelFontSize=12, titleFontSize=12),
            ),
            y=alt.Y(
                f"{y}:Q",
                title=None,
                axis=alt.Axis(labelFontSize=11),
            ),
            color=alt.Color(
                f"{x}:N",
                scale=alt.Scale(domain=list(PLATFORM_COLORS.keys()), range=list(PLATFORM_COLORS.values())),
                legend=None
            ),
            tooltip=[x, y]
        )
        .properties(
            height=260,
            title=alt.TitleParams(text=title, anchor="start", fontSize=16, offset=10),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True)
        .configure_title(fontSize=16)
    )


def platform_badge_html(p: str) -> str:
    p = str(p).strip()
    c = PLATFORM_COLORS.get(p, "#E5E7EB")
    fg = "#111827"
    return (
        f"<span style='background:{c}; color:{fg}; "
        "padding:2px 10px; border-radius:999px; font-weight:700; "
        "font-size:12px; white-space:nowrap; display:inline-flex; align-items:center; line-height:1.4;'>"
        f"{p}</span>"
    )


def type_badge_html(t: str) -> str:
    t = str(t).strip()
    c = TYPE_BADGE.get(t, {"bg": "#E5E7EB", "fg": "#111827"})
    return (
        f"<span style='background:{c['bg']}; color:{c['fg']}; "
        "padding:2px 10px; border-radius:999px; font-weight:700; "
        "font-size:12px; white-space:nowrap; display:inline-flex; align-items:center; line-height:1.4;'>"
        f"{t}</span>"
    )


# -----------------------------
# UI
# -----------------------------
# -----------------------------
# Sidebar (가장 위)
# -----------------------------
with st.sidebar:
    st.header("게임 선택")

    game = st.selectbox(
        "게임",
        ["ARCRaiders", "THEFINALS"]
    )

    data_dir = f"data/{game}"


# -----------------------------
# Main UI
# -----------------------------
st.title(f"{game} 인플루언서 전략 대시보드")

st.sidebar.markdown("### DEBUG")
st.sidebar.write(
    "GIT_SHA:",
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
)

data = load_latest_prev_streamer_softcon_and_cat(data_dir)
st.sidebar.markdown("### CAT DEBUG")
st.sidebar.write("CAT_FILE:", data["cat_filename"])
st.sidebar.write("CAT_MAX_DATE:", pd.to_datetime(data["cat_current"].get("날짜"), errors="coerce").max())

if data is None:
    st.warning("data 폴더에서 필요한 CSV를 찾지 못했습니다.\n"
               "- 스트리머_랭킹_YYYYMMDD*.csv (최소 1개)\n"
               "- 카테고리_플랫폼별_통계*.csv (최소 1개)")
    st.stop()

# 카테고리 통계(누적 1파일)에서 최신 주/전주 분리
curr_week, prev_week, cs_curr, cs_prev = split_curr_prev_weeks_from_cat_stats(data["cat_current"])
if curr_week is None:
    st.warning("카테고리_플랫폼별_통계에서 날짜를 읽지 못했습니다. '날짜' 컬럼을 확인해주세요.")
    st.stop()

# 스트리머 최신/전주 (파일명 월요일 기준으로 맞춘다고 했으니 대체로 정합)
sr_latest = data["streamer_latest"]
sr_prev = data["streamer_prev"]

# 최신 주 계산
curr_kpis, curr_platform_df, curr_top10, curr_all_streamers, curr_extra = compute_overview_metrics(sr_latest, cs_curr)

curr_pack = {
    "kpis": curr_kpis,
    "platform_df": curr_platform_df,
    "top10": curr_top10,
    "all_streamers": curr_all_streamers,
    "extra": curr_extra
}

prev_pack = None
if sr_prev is not None and prev_week is not None and len(cs_prev) > 0:
    prev_kpis, prev_platform_df, prev_top10, prev_all_streamers, prev_extra = compute_overview_metrics(sr_prev, cs_prev)
    prev_pack = {"kpis": prev_kpis, "platform_df": prev_platform_df, "top10": prev_top10, "extra": prev_extra}


# ✅ 주간 스냅샷 라벨: 스트리머/소프트콘 파일 기준
wk_latest = data.get("streamer_latest_date") or data.get("softcon_latest_date")
wk_prev   = data.get("streamer_prev_date")   or data.get("softcon_prev_date")

cols = st.columns(2)
with cols[0]:
    st.caption(f"주간 스냅샷(파일 기준): {wk_latest if wk_latest is not None else 'N/A'}")
with cols[1]:
    st.caption(f"전주 스냅샷(파일 기준): {wk_prev if wk_prev is not None else '없음'}")

# 주간 요약
st.markdown("### 주간 요약")
summary_lines = build_summary_text(curr_pack, prev_pack)
for line in summary_lines:
    st.write(line)

# KPI
st.markdown("### KPI")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
<div>
  <div style="font-size:13px;color:#6b7280;">총 뷰어십</div>
  <div style="font-size:28px;font-weight:800;">
    {kpi_with_wow(
        curr_kpis["총 뷰어십"],
        prev_pack["kpis"]["총 뷰어십"] if prev_pack else None,
        decimals=0
    )}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
<div>
  <div style="font-size:13px;color:#6b7280;">평균 시청자(카테고리)</div>
  <div style="font-size:28px;font-weight:800;">
    {kpi_with_wow(
        curr_kpis["평균 시청자(카테고리)"],
        prev_pack["kpis"]["평균 시청자(카테고리)"] if prev_pack else None,
        decimals=0
    )}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
<div>
  <div style="font-size:13px;color:#6b7280;">총 방송시간</div>
  <div style="font-size:28px;font-weight:800;">
    {kpi_with_wow(
        curr_kpis["총 방송시간"],
        prev_pack["kpis"]["총 방송시간"] if prev_pack else None,
        decimals=0
    )}
  </div>
</div>
""",
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
<div>
  <div style="font-size:13px;color:#6b7280;">가중 평균 시청자(뷰어십/시간)</div>
  <div style="font-size:28px;font-weight:800;">
    {kpi_with_wow(
        curr_kpis["가중 평균 시청자(뷰어십/시간)"],
        prev_pack["kpis"]["가중 평균 시청자(뷰어십/시간)"] if prev_pack else None,
        decimals=1
    )}
  </div>
</div>
""",
        unsafe_allow_html=True
    )


# KPI -> 플랫폼 비교 여백
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# 플랫폼 비교 차트
st.markdown("### 플랫폼 비교")
pf = curr_platform_df.copy().sort_values("뷰어십", ascending=False)
c1, c2, c3 = st.columns(3)
c1.altair_chart(bar_chart(pf, "플랫폼", "방송시간", "방송시간"), use_container_width=True)
c2.altair_chart(bar_chart(pf, "플랫폼", "평균시청자", "평균 시청자(뷰어십/방송시간)"), use_container_width=True)
c3.altair_chart(bar_chart(pf, "플랫폼", "뷰어십", "뷰어십"), use_container_width=True)

# -----------------------------
# Top 스트리머 테이블 (정렬 토글 + 배지)
# -----------------------------
st.markdown("### Top 스트리머 (Top 10)")

sort_key = st.radio(
    "정렬 기준",
    options=["뷰어십", "평균시청자"],  # 👈 디폴트가 앞에
    index=0,                            # 👈 첫 번째 = 뷰어십
    horizontal=True,
    key="top_streamer_sort"
)


sort_col = "평균시청자" if sort_key == "평균시청자" else "뷰어십"
st.caption(f"※ 정렬 기준: {sort_key}(내림차순)")

# ---------------------------
# 배지 설정 (문구/기준)  ✅ 전체 풀 기준
# ---------------------------
BADGE_TEXT = "효율 상위"       # 추천: "효율 상위" / "핵심 후보" / "고효율 운영"
LOW_HOURS_PCTL = 0.20          # 방송시간 하위 20% = '상대적으로 짧은 방송'
HIGH_AVG_PCTL = 0.80           # 평균시청자 상위 20% = '영향력 상위'

def highlight_badge_html(text: str) -> str:
    return f"""
    <span style="
        display:inline-flex;
        align-items:center;
        margin-left:8px;
        padding:2px 8px;
        border-radius:999px;
        font-size:11px;
        font-weight:800;
        background:#ECFDF5;
        color:#047857;
        border:1px solid #A7F3D0;
        line-height:1.4;
        white-space:nowrap;
    ">{text}</span>
    """

# ✅ 모집단: Top10(뷰어십) 말고 전체 풀에서 뽑아야 "짧방 고영향"이 들어온다
table = curr_all_streamers.copy()
table["기여율(%)"] = (table["기여율"] * 100).round(0)

# ✅ 배지 임계값도 전체 풀 기준
pool = curr_all_streamers.copy()
hours_threshold = pool["방송시간"].quantile(LOW_HOURS_PCTL) if pool["방송시간"].notna().any() else None
avg_threshold = pool["평균시청자"].quantile(HIGH_AVG_PCTL) if pool["평균시청자"].notna().any() else None

def is_high_eff(row) -> bool:
    if (hours_threshold is None) or (avg_threshold is None):
        return False
    return (
        pd.notna(row["방송시간"]) and pd.notna(row["평균시청자"]) and
        (row["방송시간"] <= hours_threshold) and
        (row["평균시청자"] >= avg_threshold)
    )

table["_high_eff"] = table.apply(is_high_eff, axis=1)

# ✅ 정렬 적용 후 Top10 추출 (여기서 드디어 "평균시청자 Top10"이 가능해짐)
secondary = "뷰어십" if sort_col == "평균시청자" else "평균시청자"
table = table.sort_values(
    by=[sort_col, secondary, "방송시간"],
    ascending=[False, False, False],
    na_position="last"
).head(10)

# 원하는 컬럼 순서
table = table[["플랫폼", "스트리머", "뷰어십", "기여율(%)", "방송시간", "평균시청자", "기여 타입", "_high_eff"]].copy()

# 표시용 포맷
disp = table.copy()
disp["플랫폼"] = disp["플랫폼"].map(platform_badge_html)

# 스트리머 + 배지
disp["스트리머"] = disp["스트리머"].astype(str)
disp["스트리머"] = disp.apply(
    lambda r: f'{r["스트리머"]}{highlight_badge_html(BADGE_TEXT) if r["_high_eff"] else ""}',
    axis=1
)

disp["뷰어십"] = disp["뷰어십"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
disp["기여율(%)"] = disp["기여율(%)"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
disp["방송시간"] = disp["방송시간"].map(lambda x: f"{x:>6.1f}" if pd.notna(x) else "N/A")
disp["평균시청자"] = disp["평균시청자"].map(lambda x: f"{x:>6.1f}" if pd.notna(x) else "N/A")
disp["기여 타입"] = disp["기여 타입"].map(type_badge_html)

cols = ["플랫폼", "스트리머", "뷰어십", "기여율(%)", "방송시간", "평균시청자", "기여 타입"]

rows_html = ""
for _, r in disp[cols].iterrows():
    rows_html += "<tr>" + "".join([f"<td>{r[c]}</td>" for c in cols]) + "</tr>"

# (선택) 배지 조건을 캡션으로 같이 보여주면 설득력 올라감
badge_caption = ""
if (hours_threshold is not None) and (avg_threshold is not None):
    badge_caption = f" · {BADGE_TEXT}: 방송시간 ≤ {hours_threshold:.1f}h(하위 {int(LOW_HOURS_PCTL*100)}%) & 평균시청자 ≥ {avg_threshold:.1f}(상위 {int((1-HIGH_AVG_PCTL)*100)}%)"

st.caption(f"※ 배지{badge_caption}" if badge_caption else "※ 배지: 계산 불가(데이터 부족)")

html = f"""
<style>
.table-wrap {{
  width: 100%;
  max-width: 980px;
}}

.custom-table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
}}

.custom-table th, .custom-table td {{
  border: 1px solid #e5e7eb;
  padding: 10px 12px;
  vertical-align: middle;
}}

.custom-table th {{
  background: #f9fafb;
  text-align: left;
  font-weight: 800;
}}

.custom-table td:nth-child(3),
.custom-table td:nth-child(4),
.custom-table td:nth-child(5),
.custom-table td:nth-child(6) {{
  text-align: right;
}}

.custom-table td:nth-child(1),
.custom-table td:nth-child(7) {{
  text-align: center;
}}
</style>

<div class="table-wrap">
<table class="custom-table">
  <thead>
    <tr>
      {''.join([f'<th>{c}</th>' for c in cols])}
    </tr>
  </thead>
  <tbody>
    {rows_html}
  </tbody>
</table>
</div>
"""

st.markdown(html, unsafe_allow_html=True)



# 기여 타입 정의(표 아래, 간격 포함)
st.markdown("#### 기여 타입 정의")
st.markdown(
    """
<div style="display:flex; gap:16px; align-items:center; margin-top:6px; margin-bottom:12px; flex-wrap:wrap;">

  <div style="display:flex; align-items:center; gap:6px;">
    <span style="background:#C4B5FD;color:#312E81;padding:2px 10px;border-radius:999px;
                 font-weight:700;font-size:12px;line-height:1.4;display:inline-flex;align-items:center;">
      시간형
    </span>
    <span style="font-size:12px;color:#4b5563;">
      방송시간 비중이 높은 꾸준형
    </span>
  </div>

  <div style="display:flex; align-items:center; gap:6px;">
    <span style="background:#FCA5A5;color:#7F1D1D;padding:2px 10px;border-radius:999px;
                 font-weight:700;font-size:12px;line-height:1.4;display:inline-flex;align-items:center;">
      파워형
    </span>
    <span style="font-size:12px;color:#4b5563;">
      짧은 방송에도 평균 시청자 높음
    </span>
  </div>

  <div style="display:flex; align-items:center; gap:6px;">
    <span style="background:#D1D5DB;color:#111827;padding:2px 10px;border-radius:999px;
                 font-weight:700;font-size:12px;line-height:1.4;display:inline-flex;align-items:center;">
      밸런스형
    </span>
    <span style="font-size:12px;color:#4b5563;">
      시간·시청자 모두 고르게 기여
    </span>
  </div>

</div>
""",
    unsafe_allow_html=True
)

# -----------------------------
# 캠페인 추천 (맨 아래, 기본 접힘) - 더보기 적용
# -----------------------------
st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

with st.expander("⭐캠페인 후보 탐색", expanded=False):

    # 더보기 상태 저장 키(게임별로 분리하고 싶으면 game을 포함)
    more_key = f"rec_more_{game}"

    left, right = st.columns([1.1, 2.9])

    with left:
        purpose = st.selectbox("캠페인 목적", list(PURPOSES.keys()), key="purpose_select_bottom")
        st.caption(PURPOSES[purpose]["desc"])

        pf_filter = st.multiselect(
            "플랫폼 필터",
            options=sorted(curr_platform_df["플랫폼"].unique().tolist()),
            default=sorted(curr_platform_df["플랫폼"].unique().tolist()),
            key="pf_filter_bottom"
        )

        type_filter = st.multiselect(
            "기여 타입",
            options=["시간형", "파워형", "밸런스형"],
            default=["시간형", "파워형", "밸런스형"],
            key="type_filter_bottom"
        )

        min_avg = st.number_input(
            "최소 평균 시청자",
            min_value=0,
            value=0,
            step=10,
            key="min_avg_bottom"
        )

        # ✅ 더보기 상태 초기화(없으면 False)
        if more_key not in st.session_state:
            st.session_state[more_key] = False

    # ✅ (중요) 목적/필터를 바꾸면 "더보기"는 자동으로 접히는 게 UX가 좋음
    # Streamlit은 위젯 변경 감지를 직접 받기 어렵기 때문에,
    # 추천 결과(키 파라미터) 기반으로 해시를 만들어 리셋하는 방식 사용
    state_sig = (purpose, tuple(pf_filter), tuple(type_filter), int(min_avg))
    sig_key = f"rec_sig_{game}"
    if st.session_state.get(sig_key) != state_sig:
        st.session_state[sig_key] = state_sig
        st.session_state[more_key] = False  # 리셋

    # ✅ 추천 데이터 소스: 월간 우선, 없으면 주간
    sr_rec = data.get("streamer_monthly_latest")
    sr_rec_prev = data.get("streamer_monthly_prev")
    using_monthly = (sr_rec is not None)

    if not using_monthly:
        sr_rec = sr_latest
        sr_rec_prev = sr_prev


    # -------------------------
    # 추천 계산: 20명 생성 후, 화면 표시만 10/20 토글
    # -------------------------
    curr_feat = compute_streamer_features(sr_rec)
    prev_feat = compute_streamer_features(sr_rec_prev) if sr_rec_prev is not None else None
    curr_feat = attach_prev_wow(curr_feat, prev_feat)

    rec_all = recommend_streamers(curr_feat, purpose_key=purpose, top_n=20)


    # 필터 적용
    rec_all = rec_all[rec_all["플랫폼"].isin(pf_filter)]
    rec_all = rec_all[rec_all["기여 타입"].isin(type_filter)]
    rec_all = rec_all[rec_all["평균시청자"] >= float(min_avg)]

    # 표시 개수 결정
    show_n = 20 if st.session_state[more_key] else 10
    rec = rec_all.head(show_n).copy()

    with right:
        st.caption("※ 추천점수는 목적별 가중치로 계산됩니다. 전주 데이터가 없으면 안정성 항목은 중립 처리됩니다.")

        if rec_all.empty:
            st.info("조건에 맞는 추천 후보가 없습니다. (필터/최소 평균 시청자 기준을 완화해보세요)")
        else:
            # -------------------------
            # 1) Top 3 카드 (UX 친화)
            # -------------------------
            top3 = rec_all.head(3).copy()

            def render_card(row):
                platform_badge = platform_badge_html(row["플랫폼"])
                t_badge = type_badge_html(row["기여 타입"])

                # 숫자 포맷
                score = f"{row['추천점수']:.2f}"
                vv = f"{row['뷰어십']:,.0f}"
                share = f"{row['기여율']*100:.0f}%"
                hours = f"{row['방송시간']:.1f}h"
                avg = f"{row['평균시청자']:.1f}"
                eff = f"{row['효율(뷰어십/시간)']:,.0f}" if pd.notna(row["효율(뷰어십/시간)"]) else "N/A"

                reason = str(row["추천 사유"])

                return f"""
                <div style="
                    border:1px solid #e5e7eb; border-radius:14px; padding:14px 14px;
                    background:#ffffff; box-shadow:0 1px 2px rgba(0,0,0,0.04);
                    height: 100%;
                ">
                <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                    {platform_badge}
                    <span style="font-weight:900; font-size:16px;">{row['스트리머']}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:900; font-size:14px;">Score {score}</span>
                    {t_badge}
                    </div>
                </div>

                <div style="margin-top:10px; color:#374151; font-size:13px; line-height:1.45;">
                    {reason}
                </div>

                <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; font-size:12px; color:#4b5563;">
                    <div><b>뷰어십</b> {vv}</div>
                    <div><b>기여율</b> {share}</div>
                    <div><b>방송시간</b> {hours}</div>
                    <div><b>평균시청자</b> {avg}</div>
                    <div><b>효율</b> {eff}</div>
                </div>
                </div>
                """

        st.markdown("#### 추천 Top 3")
        card_cols = st.columns(3)
        for i in range(min(3, len(top3))):
            with card_cols[i]:
                st.markdown(render_card(top3.iloc[i]), unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # -------------------------
        # 2) 표 (기본 10명 / 더보기 시 20명)
        # -------------------------
        show = rec.copy()
        show["플랫폼"] = show["플랫폼"].map(platform_badge_html)
        show["기여 타입"] = show["기여 타입"].map(type_badge_html)
        show["추천점수"] = show["추천점수"].map(lambda x: f"{x:,.2f}")
        show["뷰어십"] = show["뷰어십"].map(lambda x: f"{x:,.0f}")
        show["기여율"] = (show["기여율"] * 100).map(lambda x: f"{x:.0f}%")
        show["방송시간"] = show["방송시간"].map(lambda x: f"{x:.1f}h")
        show["평균시청자"] = show["평균시청자"].map(lambda x: f"{x:.1f}")
        show["효율(뷰어십/시간)"] = show["효율(뷰어십/시간)"].map(lambda x: f"{x:,.0f}")

        cols = ["플랫폼", "스트리머", "추천점수", "추천 사유", "뷰어십", "기여율", "방송시간", "평균시청자", "효율(뷰어십/시간)", "기여 타입"]

        rows_html = ""
        for _, r in show[cols].iterrows():
            rows_html += "<tr>" + "".join([f"<td>{r[c]}</td>" for c in cols]) + "</tr>"

        html = f"""
        <style>
        .rec-table {{
          border-collapse: collapse;
          width: 100%;
          font-size: 13px;
        }}
        .rec-table th, .rec-table td {{
          border: 1px solid #e5e7eb;
          padding: 10px 12px;
          vertical-align: top;
        }}
        .rec-table th {{
          background: #f9fafb;
          text-align: left;
          font-weight: 800;
        }}
        .rec-table td:nth-child(3),
        .rec-table td:nth-child(5),
        .rec-table td:nth-child(6),
        .rec-table td:nth-child(7),
        .rec-table td:nth-child(8),
        .rec-table td:nth-child(9) {{
          text-align: right;
          white-space: nowrap;
        }}
        .rec-table td:nth-child(1),
        .rec-table td:nth-child(10) {{
          text-align: center;
          white-space: nowrap;
        }}
        .rec-table td:nth-child(4) {{
          color: #374151;
          line-height: 1.5;
          min-width: 280px;
        }}
        </style>

        <table class="rec-table">
          <thead>
            <tr>
              {''.join([f'<th>{c}</th>' for c in cols])}
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        """
        st.markdown("#### 추천 후보 리스트")
        st.markdown(html, unsafe_allow_html=True)

        # -------------------------
        # 3) 표 아래 더보기/접기 버튼 (요청사항)
        # -------------------------
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        footer_cols = st.columns([1.2, 1.2, 7.6])
        with footer_cols[0]:
            if (not st.session_state[more_key]) and (len(rec_all) > 10):
                if st.button("더 보기 (+10)", key="btn_more_bottom"):
                    st.session_state[more_key] = True
                    st.rerun()
        with footer_cols[1]:
            if st.session_state[more_key]:
                if st.button("접기", key="btn_less_bottom"):
                    st.session_state[more_key] = False
                    st.rerun()

        with footer_cols[2]:
            if len(rec_all) > 10:
                shown = 20 if st.session_state[more_key] else 10
                st.caption(f"표시 중: {min(shown, len(rec_all))} / {len(rec_all)}명")


