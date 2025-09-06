import streamlit as st
import os
import time
from sqlalchemy import create_engine, text
import pandas as pd
from graphviz import Digraph
import sqlalchemy
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

# =========================
# App Setup & Secrets
# =========================
load_dotenv()
DEFAULT_GROQ = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="🧠 DB Engineer AI — Pro", layout="wide")

# ---------- CSS THEME INJECTION ----------
def inject_css():
    st.markdown(
        """
        <style>
        :root{
          --bg: #0b1220;
          --panel: #0f172a; /* slate-900 */
          --panel-2: #111827; /* gray-900 */
          --text: #e5e7eb; /* gray-200 */
          --muted: #9ca3af; /* gray-400 */
          --brand: #7c3aed; /* violet-600 */
          --brand-2: #22c55e; /* green-500 */
          --danger: #ef4444; /* red-500 */
          --border: rgba(148,163,184,.25); /* slate-400/25 */
          --glow: 0 10px 35px rgba(124,58,237,.25);
          --radius-xl: 16px;
          --radius-2xl: 24px;
          --shadow-1: 0 10px 24px rgba(2,6,23,.45);
          --shadow-soft: 0 6px 18px rgba(2,6,23,.28);
          --chip: rgba(124,58,237,.15);
        }

        body { background: radial-gradient(1200px 600px at 0% -10%, rgba(124,58,237,.15), transparent), var(--bg) !important;}
        .stApp { color: var(--text); }

        /* Top header bar */
        .app-header{
          display:flex; align-items:center; justify-content:space-between;
          padding:18px 22px; margin:-1rem -1rem 1rem -1rem;
          background: linear-gradient(135deg, rgba(124,58,237,.25), rgba(34,197,94,.15));
          border-bottom: 1px solid var(--border);
          box-shadow: var(--shadow-1);
        }
        .app-title{
          display:flex; gap:12px; align-items:center; font-weight:700; font-size:22px;
          letter-spacing:.2px;
        }
        .app-title .emoji{font-size:22px;filter: drop-shadow(0 4px 14px rgba(255,255,255,.2));}
        .app-caption{color:var(--muted); font-size:13px; margin-top:6px;}
        .link-row a{color:#c4b5fd; text-decoration:none; margin-left:14px;}
        .link-row a:hover{text-decoration:underline;}

        /* Cards */
        .card{
          background: linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
          border: 1px solid var(--border);
          border-radius: var(--radius-2xl);
          box-shadow: var(--shadow-soft);
          padding: 16px 18px;
        }
        .card h4{margin:0 0 8px 0;}

        /* Sidebar */
        section[data-testid="stSidebar"]{
          background: linear-gradient(180deg, rgba(17,24,39,.85), rgba(17,24,39,.6));
          border-right: 1px solid var(--border);
        }
        .sidebar-title{font-weight:700; font-size:15px; margin-bottom:6px;}
        .sidebar-sub{font-weight:600; font-size:13px; color:#cbd5e1; margin:14px 0 8px;}

        /* Inputs */
        .stTextInput > div > div > input,
        .stPassword > div > div > input,
        .stSelectbox > div > div,
        .stNumberInput input{
          background: rgba(2,6,23,.4);
          border: 1px solid var(--border);
          color: var(--text);
          border-radius: 12px;
        }
        .stSlider > div > div > div{ color: var(--brand-2) !important; }

        /* Buttons */
        .stButton>button{
          background: linear-gradient(180deg, rgba(124,58,237,.9), rgba(124,58,237,.8));
          border: 1px solid rgba(255,255,255,.08);
          color: #fff; border-radius: 12px; font-weight:600;
          box-shadow: var(--glow);
        }
        .stButton>button:hover{ filter: brightness(1.08); transform: translateY(-1px); }
        .stButton>button:active{ transform: translateY(0); }

        /* Toggles & chips */
        .tag{
          display:inline-flex; align-items:center; gap:8px;
          background: var(--chip); color:#d6bcfa; padding:6px 10px;
          border:1px solid rgba(124,58,237,.4);
          border-radius: 999px; font-size:12px; font-weight:600;
        }

        /* Chat bubbles */
        .chat-bubble{
          padding:14px 16px; border-radius: 14px;
          background: rgba(15,23,42,.6);
          border:1px solid var(--border);
          margin-bottom:10px;
        }
        .chat-bubble.user{ background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.35); }
        .chat-bubble.assistant{ background: rgba(124,58,237,.12); border-color: rgba(124,58,237,.35); }

        /* Tables */
        .stDataFrame, .stTable{
          border-radius: 14px; overflow:hidden; border:1px solid var(--border);
          box-shadow: var(--shadow-soft);
        }

        /* Expanders */
        details{
          border:1px solid var(--border) !important; border-radius: 14px !important;
          background: rgba(2,6,23,.25);
        }

        /* Code blocks */
        pre, code, .stCode{
          background: rgba(2,6,23,.55) !important;
          border: 1px solid var(--border) !important;
          border-radius: 12px !important;
        }

        /* Graphviz */
        .graph-card{ padding:12px; border-radius:16px; background:rgba(2,6,23,.3); border:1px solid var(--border); }

        /* Footer */
        .footnote{ color:#93a2b7; font-size:12px; padding:8px 0 24px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

# --- Top Header (HTML so we can style it freely)
st.markdown(
    """
    <div class="app-header">
      <div>
        <div class="app-title"><span class="emoji">🧠</span> DB Engineer Chat AI · Pro</div>
        <div class="app-caption">Query and manage your PostgreSQL with an AI SQL engineer. Safe by default.</div>
      </div>
      <div class="link-row">
        <a href="https://docs.streamlit.io/" target="_blank">Streamlit</a>
        <a href="https://docs.sqlalchemy.org/" target="_blank">SQLAlchemy</a>
        <a href="https://www.postgresql.org/docs/" target="_blank">PostgreSQL</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Session State
# =========================
ss = st.session_state
ss.setdefault("connected", False)
ss.setdefault("engine", None)
ss.setdefault("chat_history", [])
ss.setdefault("saved_queries", [])
ss.setdefault("last_sql", "")
ss.setdefault("schema_cache", {})

# =========================
# Sidebar: Connection & Settings
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔗 Connect to PostgreSQL</div>', unsafe_allow_html=True)
    host = st.text_input("Host", value=st.session_state.get("host", "localhost"))
    port = st.text_input("Port", value=st.session_state.get("port", "5432"))
    database = st.text_input("Database", value=st.session_state.get("database", "postgres"))
    username = st.text_input("Username", value=st.session_state.get("username", "postgres"))
    password = st.text_input("Password", type="password", value=st.session_state.get("password", ""))

    st.divider()
    simple_mode = st.toggle("Simple mode (hide advanced settings)", value=True)

    if not simple_mode:
        st.markdown('<div class="sidebar-sub">🤖 AI Settings</div>', unsafe_allow_html=True)
        groq_key = st.text_input("GROQ_API_KEY", type="password", value=DEFAULT_GROQ)
        model = st.selectbox(
            "Model",
            [
                "groq/llama-3.1-8b-instant",
                "groq/llama3-8b-8192",
                "groq/llama3-70b-8192",
                "groq/mixtral-8x7b-32768",
                "groq/llama-3.3-70b-versatile",
            ]
        )
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.05, 0.05)
    else:
        # sensible defaults in Simple mode
        groq_key = st.text_input("GROQ_API_KEY", type="password", value=DEFAULT_GROQ)
        model = "groq/llama-3.1-8b-instant"
        temperature = 0.05

    st.divider()
    st.markdown('<div class="sidebar-sub">🛡️ Safety & Execution</div>', unsafe_allow_html=True)
    safe_mode = st.toggle("Safe Mode (SELECT only)", value=True, help="Blocks DDL/DML.")
    auto_limit = st.number_input("Auto LIMIT for SELECT (rows)", min_value=10, max_value=10000, value=200, step=10)
    max_seconds = st.number_input("Query timeout (seconds)", min_value=1, max_value=600, value=60)

    st.divider()
    cA, cB = st.columns(2)
    with cA:
        if st.button("Connect", use_container_width=True):
            try:
                dsn = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
                engine = create_engine(dsn, pool_pre_ping=True, pool_recycle=180)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                ss.connected, ss.engine = True, engine
                ss.host, ss.port, ss.database, ss.username, ss.password = host, port, database, username, password
                os.environ["GROQ_API_KEY"] = (groq_key or "").strip()
                st.success("✅ Connected to PostgreSQL!")
            except Exception as e:
                ss.connected, ss.engine = False, None
                st.error(f"❌ Connection failed: {e}")
    with cB:
        if st.button("Disconnect", use_container_width=True):
            try:
                if ss.engine:
                    ss.engine.dispose()
                ss.connected, ss.engine = False, None
                st.success("🔌 Disconnected!")
            except Exception as e:
                st.error(f"❌ Error while disconnecting: {e}")

# =========================
# Helpers
# =========================
def build_schema_summary(engine, schema: str = "public", max_chars: int = 14000) -> str:
    try:
        insp = sqlalchemy.inspect(engine)
        tables = insp.get_table_names(schema=schema)
        parts = [f"Schema: {schema}"]
        for t in tables:
            cols = insp.get_columns(t, schema=schema)
            col_s = ", ".join([f"{c['name']}:{str(c['type']).split('(')[0]}" for c in cols])
            parts.append(f"- {t} => {col_s}")
        for t in tables:
            try:
                for fk in insp.get_foreign_keys(t, schema=schema):
                    if fk and fk.get("referred_table"):
                        parts.append(
                            f"FK: {t}({', '.join(fk.get('constrained_columns', []))}) → {fk.get('referred_table')}({', '.join(fk.get('referred_columns', []))})"
                        )
            except Exception:
                pass
        txt = "\n".join(parts)
        return txt[:max_chars]
    except Exception as e:
        return f"[Introspection failed: {e}]"

def generate_schema_graph(engine, schema: str = "public") -> Digraph:
    dot = Digraph()
    dot.attr(rankdir="LR")
    insp = sqlalchemy.inspect(engine)
    tables = insp.get_table_names(schema=schema)
    if not tables:
        return Digraph()
    for table in tables:
        dot.node(table, label=f"{table}", shape="box")
        for col in insp.get_columns(table, schema=schema):
            col_id = f"{table}.{col['name']}"
            col_type = str(col['type']).split('(')[0]
            dot.node(col_id, f"{col['name']} ({col_type})", shape="ellipse")
            dot.edge(table, col_id)
    for table in tables:
        for fk in insp.get_foreign_keys(table, schema=schema):
            if fk and fk.get("referred_table"):
                dot.edge(table, fk["referred_table"], label="FK")
    return dot

def enforce_safe_mode(sql: str, safe: bool) -> tuple[bool, str]:
    s = sql.strip().lower()
    allowed_heads = ("select", "explain") if safe else (
        "select", "insert", "update", "delete", "create", "drop", "alter", "grant", "revoke", "truncate", "explain"
    )
    if not any(s.startswith(h) for h in allowed_heads):
        return False, "Blocked by Safe Mode. Only SELECT/EXPLAIN are allowed."
    if ";" in s[:-1]:
        return False, "Multiple statements detected. Return exactly one statement."
    return True, ""

def ensure_select_limit(sql: str, max_rows: int) -> str:
    s = sql.strip()
    if s.lower().startswith("select") and " limit " not in s.lower():
        return s.rstrip(";") + f" LIMIT {max_rows};"
    if not s.endswith(";"):
        s += ";"
    return s

def _crew_result_to_str(result) -> str:
    """CrewAI sometimes returns a str, sometimes an object with .raw"""
    if isinstance(result, str):
        return result
    raw = getattr(result, "raw", None)
    if isinstance(raw, str):
        return raw
    return str(result).strip()

# =========================
# Main Layout
# =========================
left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="card"><h4>🧩 Chat</h4>', unsafe_allow_html=True)

    # Render chat history with “bubbles”
    for role, msg in ss.chat_history:
        bubble_cls = "user" if role == "user" else "assistant"
        st.markdown(f'<div class="chat-bubble {bubble_cls}">', unsafe_allow_html=True)
        if msg.strip().lower().startswith(("select", "explain", "insert", "update", "delete", "create", "drop", "alter", "grant", "revoke", "truncate")):
            st.code(msg, language="sql")
        else:
            st.markdown(msg)
        st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask anything about your database (e.g., 'list tables', 'top 50 orders by total').")

    st.markdown('</div>', unsafe_allow_html=True)  # close .card

    if prompt and not ss.connected:
        st.warning("Please connect to the database first from the sidebar.")

    if prompt and ss.connected:
        ss.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Analyzing schema…"):
            schema_text = build_schema_summary(ss.engine, schema="public")

        sys_backstory = (
            "You are a battle-tested PostgreSQL principal engineer.\n"
            "Your job: translate any natural language into ONE valid PostgreSQL SQL statement.\n"
            "Rules: one statement only; semicolon at end; uppercase keywords; use PostgreSQL only."
        )
        schema_context = f"\nDatabase schema snapshot (public):\n{schema_text}\n"

        agent = Agent(
            role="PostgreSQL Expert",
            goal="Generate one correct PostgreSQL SQL statement for the user's request.",
            backstory=sys_backstory + schema_context,
            verbose=False,
            llm=model,
            temperature=temperature,
        )

        task = Task(
            description=(f"Natural language:\n{prompt}\n\nReturn exactly one valid PostgreSQL SQL statement terminated by a semicolon."),
            expected_output="One valid SQL statement ending with a semicolon.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        t0 = time.time()
        try:
            os.environ["GROQ_API_KEY"] = (st.session_state.get("GROQ_API_KEY") or "").strip() if "GROQ_API_KEY" in st.session_state else (DEFAULT_GROQ or "")
            if not os.environ.get("GROQ_API_KEY") and "groq_key" in locals():
                os.environ["GROQ_API_KEY"] = (groq_key or "").strip()

            result_obj = crew.kickoff()
            sql_query = _crew_result_to_str(result_obj).strip()
            if not sql_query.endswith(";"):
                sql_query += ";"

            allowed, reason = enforce_safe_mode(sql_query, safe_mode)
            if not allowed:
                with st.chat_message("assistant"):
                    st.error(f"🚫 {reason}")
                    st.code(sql_query, language="sql")
                ss.chat_history.append(("assistant", sql_query))
                st.stop()

            sql_query = ensure_select_limit(sql_query, auto_limit)

            with st.chat_message("assistant"):
                st.markdown("**Proposed SQL:**")
                edited = st.text_area("Review & edit before execution", value=sql_query, height=150)
                run_cols = st.columns(3)
                do_explain = run_cols[0].button("EXPLAIN", use_container_width=True)
                do_run = run_cols[1].button("Run", type="primary", use_container_width=True)
                save_it = run_cols[2].button("Save query", use_container_width=True)

            if save_it:
                ss.saved_queries.append({"prompt": prompt, "sql": edited})
                st.toast("Saved.")

            if do_explain or do_run:
                sql_to_exec = edited.strip()
                allowed, reason = enforce_safe_mode(sql_to_exec, safe_mode)
                if not allowed:
                    st.error(f"🚫 {reason}")
                else:
                    try:
                        with ss.engine.connect() as conn:
                            conn.execute(text(f"SET statement_timeout = {int(max_seconds*1000)}"))

                            if do_explain and not sql_to_exec.lower().startswith("explain"):
                                sql_to_exec = "EXPLAIN " + sql_to_exec.rstrip(";") + ";"

                            q0 = time.time()
                            res = conn.execute(text(sql_to_exec))
                            conn.commit()
                            elapsed = time.time() - q0

                            head = sql_to_exec.strip().lower()
                            if head.startswith(("select", "explain")):
                                try:
                                    rows = res.fetchall()
                                    cols = list(res.keys()) if hasattr(res, "keys") else []
                                    if rows:
                                        df = pd.DataFrame(rows, columns=cols)
                                        st.caption(f"⏱️ {elapsed:.2f}s")
                                        st.dataframe(df, use_container_width=True)
                                        csv = df.to_csv(index=False).encode("utf-8")
                                        st.download_button("📥 Download CSV", csv, "query_results.csv")
                                    else:
                                        st.info("ℹ️ No rows returned.")
                                except Exception:
                                    st.success("✅ Executed.")
                            else:
                                st.success(f"✅ Executed in {elapsed:.2f}s.")

                            if any(k in head for k in ["create table", "drop table", "alter table", "create database", "drop database", "create schema", "drop schema"]):
                                st.info("Schema changed. See diagram on the right →")

                    except Exception as e:
                        st.error(f"❌ SQL execution error: {e}")

            ss.last_sql = edited if 'edited' in locals() else sql_query
            ss.chat_history.append(("assistant", ss.last_sql))

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"❌ Error generating SQL: {e}")

        finally:
            gen_time = time.time() - t0
            st.caption(f"🧠 Generation time: {gen_time:.2f}s")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Schema & Tools")

    if ss.connected:
        schema_choice = st.selectbox("Schema", options=["public"], index=0)
        st.markdown("**Database Schema Diagram**")
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)
        try:
            graph = generate_schema_graph(ss.engine, schema_choice)
            st.graphviz_chart(graph, use_container_width=True)
        except Exception as e:
            st.error(f"Diagram error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("**Quick Introspection**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("List tables", use_container_width=True):
                try:
                    with ss.engine.connect() as conn:
                        res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1"))
                        df = pd.DataFrame(res.fetchall(), columns=["table_name"])
                        st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(e)
        with c2:
            tbl = st.text_input("Preview table", placeholder="e.g. users")
            if st.button("Preview 50 rows", use_container_width=True) and tbl:
                try:
                    with ss.engine.connect() as conn:
                        res = conn.execute(text(f"SELECT * FROM {tbl} LIMIT 50"))
                        df = pd.DataFrame(res.fetchall(), columns=res.keys())
                        st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(e)

        if not simple_mode:
            st.divider()
            st.markdown("**Saved Queries**")
            if ss.saved_queries:
                for i, q in enumerate(ss.saved_queries[::-1]):
                    with st.expander(f"{q['prompt']}"):
                        st.code(q["sql"], language="sql")
                        colA, colB = st.columns(2)
                        if colA.button("Copy to editor", key=f"copy{i}"):
                            ss.chat_history.append(("assistant", q["sql"]))
                            st.toast("Copied to chat.")
                        if colB.button("Delete", key=f"del{i}"):
                            idx = len(ss.saved_queries) - 1 - i
                            del ss.saved_queries[idx]
                            st.experimental_rerun()
            else:
                st.caption("No saved queries yet.")
    else:
        st.info("Connect to a database to view schema and tools.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Footer & Cleanup
# =========================
st.markdown('<div class="footnote">Tip: Safe Mode adds protection. Disable only when you know exactly what you\'re doing.</div>', unsafe_allow_html=True)

def cleanup():
    if ss.connected and ss.engine:
        try:
            ss.engine.dispose()
        except Exception:
            pass

import atexit
atexit.register(cleanup)
