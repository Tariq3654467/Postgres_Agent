DB Engineer Chat AI · Pro (Streamlit)

Query and manage your PostgreSQL database using natural language. The app converts your prompts into one safe PostgreSQL statement, lets you review/edit, then EXPLAIN or run it. Includes schema introspection and a live diagram.

Features

Chat-to-SQL via CrewAI + Groq (LiteLLM routing under the hood)

Safety first: Safe Mode (SELECT/EXPLAIN only), single-statement guard, auto-LIMIT, statement timeout

Schema aware: quick schema snapshot for grounding + Graphviz diagram

Editable SQL: review, edit, EXPLAIN, run, and download results as CSV

Saved queries: save, recall, copy to editor

Modern UI: custom CSS (glass cards, soft shadows), optional Simple Mode to hide advanced settings

Quick Start (Local)
1) Requirements

Python 3.10+

PostgreSQL you can reach from your machine

System Graphviz (for the schema diagram)

Install Graphviz:

Ubuntu/Debian: sudo apt-get install graphviz

macOS (Homebrew): brew install graphviz

Windows: install from the official Graphviz site and add to PATH

2) Clone and setup
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


requirements.txt

streamlit
sqlalchemy
psycopg2-binary
pandas
graphviz
crewai
python-dotenv

3) Configure secrets

Create a .env file (or use Streamlit secrets on cloud):

GROQ_API_KEY=gsk_your_real_groq_key_here


Your key must start with gsk_…. OpenAI keys (sk-…) won’t work.

4) Run
streamlit run app.py


Open the app in the browser (Streamlit shows the URL).

Using the App

Connect to PostgreSQL from the sidebar

Host, Port, Database, Username, Password

(Optional) paste your GROQ_API_KEY if not in .env

Click Connect

Ask in Chat
Examples:

“list tables”

“top 50 orders by total”

“columns and types in users”

“describe table customers”

“count users per country order by count desc”

Review the SQL

App returns one SQL statement; you can edit before execution

Click EXPLAIN to check the plan or Run to execute

Browse schema

Right panel shows diagram, quick introspection, and optional saved queries

Keep Safe Mode ON unless you must write/alter data. It blocks DDL/DML and multiple statements.

Environment & Models

The app expects Groq via LiteLLM. Good model IDs:

groq/llama-3.1-8b-instant (default)

groq/llama3-8b-8192

groq/llama3-70b-8192

groq/mixtral-8x7b-32768

groq/llama-3.3-70b-versatile

Set the key:

.env → GROQ_API_KEY=gsk_...

or paste it in the sidebar (app uses the sidebar value on Connect)
