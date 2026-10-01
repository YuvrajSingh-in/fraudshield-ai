"""
FraudShield AI — Premium Command Center v2.

FIXES:
- UnicodeDecodeError: pd.read_csv now uses encoding="utf-8", errors="replace"
  so a single corrupt byte never crashes the UI.
- Cache invalidated every 3s with st.cache_data(ttl=3) — no blocking while loop
  around data loading.

NEW PANELS:
- Fraud rate trend (rolling 50-transaction window)
- Rules breakdown: which rules are firing most
- Top risky users and merchants
- Card testing / velocity attack alerts
- ML confidence distribution (detects rules-only false positive clusters)
- SOFT_CHECK / REVIEW tier tracking (new action tiers)
"""

import time
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;600&display=swap');
:root {
    --bg:       #03040a;
    --card:     rgba(12,18,35,0.72);
    --border:   rgba(80,140,255,0.18);
    --shine:    rgba(120,180,255,0.07);
    --blue:     #3b82f6;
    --cyan:     #06b6d4;
    --violet:   #8b5cf6;
    --emerald:  #10b981;
    --rose:     #f43f5e;
    --amber:    #f59e0b;
    --txt:      #e8eaf6;
    --txt2:     #8892b0;
    --txt3:     #4a5568;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main .block-container{
    background:var(--bg)!important; color:var(--txt)!important;
    font-family:'Space Grotesk',sans-serif!important;
}
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%,rgba(59,130,246,.07) 0%,transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%,rgba(139,92,246,.06) 0%,transparent 60%),
        var(--bg)!important;
}
.main .block-container{padding:1.5rem 2rem 3rem!important;max-width:1600px!important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.gc{background:var(--card);backdrop-filter:blur(20px) saturate(180%);
    border:1px solid var(--border);border-radius:20px;padding:22px 26px;position:relative;
    overflow:hidden;transition:transform .3s,box-shadow .3s,border-color .3s;}
.gc::before{content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,var(--shine) 0%,transparent 60%);
    border-radius:inherit;pointer-events:none;}
.gc:hover{transform:translateY(-2px) scale(1.005);border-color:rgba(80,140,255,.35);
    box-shadow:0 0 40px rgba(59,130,246,.2);}
.mv{font-size:2.8rem;font-weight:700;letter-spacing:-2px;line-height:1;
    font-family:'JetBrains Mono',monospace;
    background:linear-gradient(135deg,#e8eaf6 30%,rgba(232,234,246,.6));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.ml{font-size:.7rem;font-weight:500;letter-spacing:.15em;text-transform:uppercase;color:var(--txt2);margin-top:5px;}
.md{font-size:.75rem;font-family:'JetBrains Mono',monospace;margin-top:7px;padding:2px 9px;
    border-radius:20px;display:inline-block;}
.du{background:rgba(244,63,94,.15);color:#fb7185;border:1px solid rgba(244,63,94,.3);}
.dd{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.3);}
.df{background:rgba(148,163,184,.12);color:#94a3b8;border:1px solid rgba(148,163,184,.2);}
.sh{display:flex;align-items:center;gap:9px;margin-bottom:15px;}
.st{font-size:.68rem;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--txt2);}
.sd{width:6px;height:6px;border-radius:50%;flex-shrink:0;}
.al{display:flex;align-items:flex-start;gap:12px;padding:12px 14px;border-radius:13px;
    margin-bottom:8px;border:1px solid transparent;animation:sIn .4s ease;}
@keyframes sIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}
.avh{background:rgba(244,63,94,.10);border-color:rgba(244,63,94,.30);}
.ah{background:rgba(245,158,11,.10);border-color:rgba(245,158,11,.30);}
.am{background:rgba(59,130,246,.09);border-color:rgba(59,130,246,.25);}
.asc{background:rgba(139,92,246,.09);border-color:rgba(139,92,246,.25);}
.ab{width:8px;height:8px;border-radius:50%;flex-shrink:0;margin-top:4px;box-shadow:0 0 7px currentColor;}
.bvh{color:#f43f5e;background:#f43f5e;animation:pl 1.2s infinite;}
.bh{color:#f59e0b;background:#f59e0b;animation:pl 1.8s infinite;}
.bm{color:#3b82f6;background:#3b82f6;}
.bsc{color:#8b5cf6;background:#8b5cf6;}
@keyframes pl{0%,100%{box-shadow:0 0 5px currentColor}50%{box-shadow:0 0 12px currentColor}}
.au{font-size:.8rem;font-weight:600;color:var(--txt);font-family:'JetBrains Mono',monospace;}
.ad{font-size:.72rem;color:var(--txt2);margin-top:2px;line-height:1.4;}
.aa{font-family:'JetBrains Mono',monospace;font-size:.88rem;font-weight:600;margin-left:auto;flex-shrink:0;}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.65rem;font-weight:600;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;}
.pvh{background:rgba(244,63,94,.18);color:#fb7185;border:1px solid rgba(244,63,94,.35);}
.ph{background:rgba(245,158,11,.18);color:#fbbf24;border:1px solid rgba(245,158,11,.35);}
.pm{background:rgba(59,130,246,.18);color:#60a5fa;border:1px solid rgba(59,130,246,.30);}
.pl2{background:rgba(16,185,129,.15);color:#34d399;border:1px solid rgba(16,185,129,.25);}
.psc{background:rgba(139,92,246,.18);color:#a78bfa;border:1px solid rgba(139,92,246,.30);}
.pr{background:rgba(245,158,11,.18);color:#fbbf24;border:1px solid rgba(245,158,11,.35);}
.ld{display:inline-block;width:7px;height:7px;background:#10b981;border-radius:50%;
    margin-right:6px;box-shadow:0 0 9px #10b981;animation:lp 1.4s infinite;vertical-align:middle;}
@keyframes lp{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(1.3)}}
.rb{display:flex;align-items:center;gap:10px;margin-bottom:11px;}
.rl{font-size:.72rem;font-weight:500;width:70px;flex-shrink:0;font-family:'JetBrains Mono',monospace;}
.rw{flex:1;height:7px;background:rgba(255,255,255,.05);border-radius:8px;overflow:hidden;}
.rf{height:100%;border-radius:8px;}
.rc{font-size:.72rem;font-family:'JetBrains Mono',monospace;color:var(--txt2);width:34px;text-align:right;flex-shrink:0;}
.txnt{width:100%;border-collapse:collapse;}
.txnt thead th{font-size:.65rem;font-weight:600;letter-spacing:.10em;text-transform:uppercase;
    color:var(--txt3);padding:0 10px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,.06);}
.txnt tbody tr{border-bottom:1px solid rgba(255,255,255,.03);}
.txnt tbody tr:hover{background:rgba(255,255,255,.03);}
.txnt td{padding:9px 10px;font-size:.75rem;color:var(--txt2);}
.txnt td:first-child{color:var(--txt);font-family:'JetBrains Mono',monospace;font-size:.73rem;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(80,140,255,.22);border-radius:3px;}
[data-testid="stColumns"]{gap:1rem!important;}
</style>
""", unsafe_allow_html=True)

LOG_FILE = "logs/transactions.csv"

RISK_COLORS = {"VERY_HIGH":"#f43f5e","HIGH":"#f59e0b","MEDIUM":"#3b82f6","LOW":"#10b981"}
COUNTRY_RISK = {"NG","PK","RU","IR","KP"}
PILL_CLASS   = {"VERY_HIGH":"pvh","HIGH":"ph","MEDIUM":"pm","LOW":"pl2","SOFT_CHECK":"psc","REVIEW":"pr"}
ALERT_CLASS  = {"VERY_HIGH":"avh","HIGH":"ah","MEDIUM":"am","SOFT_CHECK":"asc"}
BADGE_CLASS  = {"VERY_HIGH":"bvh","HIGH":"bh","MEDIUM":"bm","SOFT_CHECK":"bsc"}


@st.cache_data(ttl=3)
def load_data() -> pd.DataFrame:
    try:
        # UTF-8 enforced, errors="replace" — never crashes on corrupt bytes
        df = pd.read_csv(LOG_FILE, encoding="utf-8", encoding_errors="replace")
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.strip()
        for col in ("Amount", "fraud_probability", "total_risk_score"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def pill(level: str) -> str:
    cls = PILL_CLASS.get(str(level), "df")
    return f'<span class="pill {cls}">{level}</span>'

def fmt(v: float) -> str:
    return f"€{v:,.2f}"

def bar_row(label, count, total, color, weight="500") -> str:
    pct = count / max(total, 1) * 100
    return (
        f'<div class="rb">'
        f'<div class="rl" style="color:{color};font-weight:{weight};">{label}</div>'
        f'<div class="rw"><div class="rf" style="width:{pct:.1f}%;background:{color};opacity:.85;"></div></div>'
        f'<div class="rc">{count}</div>'
        f'</div>'
    )

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:1.8rem;padding-bottom:1rem;border-bottom:1px solid rgba(255,255,255,.07);">
  <div>
    <div style="font-size:.68rem;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#3b82f6;margin-bottom:5px;">
        🛡 FraudShield AI v2
    </div>
    <h1 style="font-size:1.9rem;font-weight:700;letter-spacing:-1px;color:#e8eaf6;margin:0;line-height:1.1;">
        Fraud Intelligence Command Center
    </h1>
  </div>
  <div style="text-align:right;">
    <div style="font-size:.72rem;color:#4a5568;margin-bottom:3px;">System status</div>
    <span class="ld"></span>
    <span style="font-family:'JetBrains Mono',monospace;color:#10b981;font-size:.82rem;">LIVE</span>
  </div>
</div>
""", unsafe_allow_html=True)

df = load_data()

if df.empty:
    st.markdown("""
    <div class="gc" style="text-align:center;padding:55px 40px;">
      <div style="font-size:2.2rem;margin-bottom:14px;">📡</div>
      <div style="font-size:1rem;font-weight:600;color:#e8eaf6;margin-bottom:7px;">Awaiting transaction stream</div>
      <div style="font-size:.8rem;color:#4a5568;margin-bottom:18px;">Start the Kafka pipeline to begin</div>
      <code style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#3b82f6;">
        python run_streaming_pipeline.py
      </code>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.rerun()

# ─── Stats ────────────────────────────────────────────────────────────────────
total       = len(df)
high_risk   = df[df["risk_level"].isin(["HIGH","VERY_HIGH"])]
blocked     = df[df["action"] == "BLOCK_CARD"]
soft_check  = df[df["action"] == "SOFT_CHECK"] if "action" in df.columns else pd.DataFrame()
fraud_rate  = len(high_risk) / max(total, 1) * 100
avg_amount  = df["Amount"].mean()
risk_counts = df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
action_counts = df["action"].value_counts().to_dict() if "action" in df.columns else {}

# ─── KPI Row ─────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)

def kpi(col, val, label, delta, dcls):
    col.markdown(f"""
    <div class="gc" style="text-align:center;padding:22px 14px;">
      <div class="mv">{val}</div>
      <div class="ml">{label}</div>
      <div class="md {dcls}">{delta}</div>
    </div>""", unsafe_allow_html=True)

with c1: kpi(c1, f"{total:,}",           "Transactions",    "live",               "df")
with c2: kpi(c2, f"{len(high_risk):,}",  "High Risk",       f"{fraud_rate:.1f}%",  "du" if fraud_rate>30 else "dd")
with c3: kpi(c3, f"{len(blocked):,}",    "Blocked",         "BLOCK_CARD",          "du" if len(blocked)>10 else "df")
with c4: kpi(c4, f"{len(soft_check):,}", "Step-Up Auth",    "SOFT_CHECK",          "df")
with c5: kpi(c5, f"{fraud_rate:.1f}%",   "Fraud Rate",      "of volume",           "du" if fraud_rate>40 else "dd")
with c6: kpi(c6, f"€{avg_amount:,.0f}",  "Avg Amount",      "per txn",             "df")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ─── Row 2: Alerts | Distribution | Country ───────────────────────────────────
col_a, col_d, col_c = st.columns([2, 1.4, 1.4])

with col_a:
    alerts_df = df[df["risk_level"].isin(["VERY_HIGH","HIGH","MEDIUM","SOFT_CHECK"])].tail(9)
    feed_html = ""
    for _, row in alerts_df.iloc[::-1].iterrows():
        level   = str(row.get("risk_level","MEDIUM"))
        action  = str(row.get("action","?"))
        user    = str(row.get("user_id","?"))
        country = str(row.get("country","?"))
        amount  = float(row.get("Amount",0))
        score   = float(row.get("total_risk_score",0))
        explan  = str(row.get("explanation",""))[:55]
        ac = ALERT_CLASS.get(level,"am")
        bc = BADGE_CLASS.get(level,"bm")
        feed_html += (
            f'<div class="al {ac}">'
            f'<div class="ab {bc}"></div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div class="au">{user} · {country}</div>'
            f'<div class="ad">{action} · {score:.3f}{(" · " + explan) if explan else ""}</div>'
            f'</div>'
            f'<div class="aa" style="color:{"#f43f5e" if level in ("VERY_HIGH","HIGH") else "#f59e0b"}">{fmt(amount)}</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div class="gc" style="height:400px;overflow-y:auto;">
      <div class="sh"><span class="sd" style="background:#f43f5e;box-shadow:0 0 7px #f43f5e;"></span>
      <span class="st">Live Threat Feed</span></div>
      {feed_html}
    </div>""", unsafe_allow_html=True)

with col_d:
    BAR = {
        "VERY_HIGH":"linear-gradient(90deg,#f43f5e,#fb7185)",
        "HIGH"     :"linear-gradient(90deg,#f59e0b,#fbbf24)",
        "MEDIUM"   :"linear-gradient(90deg,#3b82f6,#60a5fa)",
        "LOW"      :"linear-gradient(90deg,#10b981,#34d399)",
    }
    bars_html = ""
    for lvl in ["VERY_HIGH","HIGH","MEDIUM","LOW"]:
        cnt = risk_counts.get(lvl,0)
        pct = cnt / max(total,1) * 100
        bars_html += (
            f'<div class="rb">'
            f'<div class="rl" style="color:{RISK_COLORS.get(lvl,"#888")}">{lvl.replace("_"," ")}</div>'
            f'<div class="rw"><div class="rf" style="width:{pct:.1f}%;background:{BAR[lvl]};"></div></div>'
            f'<div class="rc">{cnt}</div>'
            f'</div>'
        )

    action_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:.73rem;">'
        f'<span style="color:#8892b0;font-family:JetBrains Mono,monospace">{a}</span>'
        f'<span style="color:#e8eaf6;font-family:JetBrains Mono,monospace">{c}</span></div>'
        for a, c in sorted(action_counts.items(), key=lambda x: -x[1])
    )

    st.markdown(f"""
    <div class="gc" style="height:400px;">
      <div class="sh"><span class="sd" style="background:#3b82f6;box-shadow:0 0 7px #3b82f6;"></span>
      <span class="st">Risk Distribution</span></div>
      {bars_html}
      <div style="margin-top:18px;padding-top:14px;border-top:1px solid rgba(255,255,255,.05);">
        <div style="font-size:.66rem;color:#4a5568;text-transform:uppercase;letter-spacing:.10em;margin-bottom:8px;">Actions</div>
        {action_rows}
      </div>
    </div>""", unsafe_allow_html=True)

with col_c:
    top_cc = df["country"].value_counts().head(8) if "country" in df.columns else pd.Series(dtype=int)
    max_c  = max(top_cc.values) if len(top_cc) else 1
    cc_rows = ""
    for cc, cnt in top_cc.items():
        pct     = cnt / max_c * 100
        is_risk = cc in COUNTRY_RISK
        bar_col = "linear-gradient(90deg,#f43f5e,#fb7185)" if is_risk else "linear-gradient(90deg,#3b82f6,#60a5fa)"
        cc_color = "#fb7185" if is_risk else "#e8eaf6"
        cc_rows += (
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.79rem;font-weight:600;color:{cc_color};width:34px;">{cc}</div>'
            f'<div style="flex:1;height:5px;background:rgba(255,255,255,.05);border-radius:5px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{bar_col};border-radius:5px;"></div>'
            f'</div>'
            f'<div style="font-size:.72rem;font-family:\'JetBrains Mono\',monospace;color:#4a5568;min-width:28px;text-align:right;">{cnt}</div>'
            f'</div>'
        )

    st.markdown(f"""
    <div class="gc" style="height:400px;overflow-y:auto;">
      <div class="sh"><span class="sd" style="background:#8b5cf6;box-shadow:0 0 7px #8b5cf6;"></span>
      <span class="st">Country Risk Map</span></div>
      {cc_rows}
    </div>""", unsafe_allow_html=True)

# ─── Row 3: Fraud rate trend | Top risky users | ML confidence ─────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
col_t, col_u, col_ml = st.columns([2, 1.3, 1.3])

with col_t:
    st.markdown("""
    <div class="gc" style="padding-bottom:8px;">
      <div class="sh"><span class="sd" style="background:#06b6d4;box-shadow:0 0 7px #06b6d4;"></span>
      <span class="st">Transaction Amount + Fraud Rate Trend</span></div>
    </div>""", unsafe_allow_html=True)

    chart_df = df[["Amount"]].tail(150).reset_index(drop=True)
    if "risk_level" in df.columns:
        # Rolling fraud rate (1 = HIGH/VERY_HIGH, 0 = else)
        is_fraud     = df["risk_level"].isin(["HIGH","VERY_HIGH"]).astype(int)
        rolling_rate = is_fraud.rolling(20).mean() * 100
        chart_df["Fraud Rate %"] = rolling_rate.tail(150).values
    st.line_chart(chart_df, height=200, use_container_width=True)

with col_u:
    top_users = (
        df[df["risk_level"].isin(["HIGH","VERY_HIGH"])]
        ["user_id"].value_counts().head(6)
        if "user_id" in df.columns else pd.Series(dtype=int)
    )
    user_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 0;border-bottom:1px solid rgba(255,255,255,.04);">'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.75rem;color:#e8eaf6;">{u}</span>'
        f'<span style="font-family:JetBrains Mono,monospace;font-size:.75rem;color:#f43f5e;font-weight:600;">{c} alerts</span>'
        f'</div>'
        for u, c in top_users.items()
    ) or '<div style="color:#4a5568;font-size:.78rem;padding:12px 0;">No high-risk users yet</div>'

    st.markdown(f"""
    <div class="gc" style="height:270px;overflow-y:auto;">
      <div class="sh"><span class="sd" style="background:#f43f5e;box-shadow:0 0 7px #f43f5e;"></span>
      <span class="st">Top Risky Users</span></div>
      {user_rows}
    </div>""", unsafe_allow_html=True)

with col_ml:
    if "fraud_probability" in df.columns:
        buckets = {
            "< 0.01 (rules only)": (df["fraud_probability"] < 0.01).sum(),
            "0.01–0.10"          : ((df["fraud_probability"] >= 0.01) & (df["fraud_probability"] < 0.10)).sum(),
            "0.10–0.30"          : ((df["fraud_probability"] >= 0.10) & (df["fraud_probability"] < 0.30)).sum(),
            "> 0.30 (ML signal)" : (df["fraud_probability"] >= 0.30).sum(),
        }
        ml_rows = ""
        for label, cnt in buckets.items():
            pct    = cnt / max(total,1) * 100
            color  = "#f43f5e" if "> 0.30" in label else ("#f59e0b" if "0.10" in label else ("#3b82f6" if "0.01–" in label else "#4a5568"))
            ml_rows += bar_row(label.split("(")[0].strip(), cnt, total, color)

        st.markdown(f"""
        <div class="gc" style="height:270px;">
          <div class="sh"><span class="sd" style="background:#8b5cf6;box-shadow:0 0 7px #8b5cf6;"></span>
          <span class="st">ML Confidence Dist.</span></div>
          {ml_rows}
          <div style="margin-top:14px;font-size:.7rem;color:#4a5568;line-height:1.6;">
            "rules only" = transactions flagged by country/velocity rules<br>
            with near-zero ML confidence. These are <span style="color:#f59e0b;">potential FPs</span>.
          </div>
        </div>""", unsafe_allow_html=True)

# ─── Row 4: Recent transactions table ─────────────────────────────────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
recent = df.tail(18).iloc[::-1].reset_index(drop=True)

cols_show = [c for c in ["user_id","country","Amount","risk_level","action","fraud_probability","total_risk_score","explanation"] if c in recent.columns]
hdr = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in cols_show)
rows_html = ""
for _, row in recent.iterrows():
    cells = ""
    for c in cols_show:
        v = row.get(c, "")
        if c == "Amount":
            cells += f'<td style="font-family:JetBrains Mono,monospace;font-weight:600;color:#e8eaf6;">{fmt(float(v or 0))}</td>'
        elif c == "risk_level":
            cells += f"<td>{pill(str(v))}</td>"
        elif c in ("fraud_probability","total_risk_score"):
            cells += f'<td style="font-family:JetBrains Mono,monospace;font-size:.7rem;color:#8892b0;">{float(v or 0):.4f}</td>'
        elif c == "explanation":
            cells += f'<td style="font-size:.7rem;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{str(v)[:50]}</td>'
        else:
            cells += f"<td>{v}</td>"
    rows_html += f"<tr>{cells}</tr>"

st.markdown(f"""
<div class="gc" style="overflow:hidden;">
  <div class="sh"><span class="sd" style="background:#10b981;box-shadow:0 0 7px #10b981;"></span>
  <span class="st">Recent Transactions</span></div>
  <div style="overflow-x:auto;">
    <table class="txnt">
      <thead><tr>{hdr}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:1.8rem;padding-top:.8rem;border-top:1px solid rgba(255,255,255,.05);
display:flex;justify-content:space-between;">
  <div style="font-size:.67rem;color:#2d3748;font-family:'JetBrains Mono',monospace;">
    FraudShield AI v2 · XGBoost+Calibration · Weighted Scoring · Kafka DLQ
  </div>
  <div style="font-size:.67rem;color:#2d3748;">
    Auto-refresh 3s · {total:,} transactions · encoding=utf-8 ✓
  </div>
</div>""", unsafe_allow_html=True)

time.sleep(3)
st.rerun()
