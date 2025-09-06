# 🚀 Postgres Agent – DB Engineer Chat AI (Pro)

Query, visualize, and manage your PostgreSQL database with natural language.  
Built on Streamlit, powered by CrewAI + Groq, with a modern UI and safety-first design.

---

## 🌟 Features
- **AI Chat-to-SQL**: Natural language prompts converted to safe SQL (CrewAI + Groq, via LiteLLM).
- **Safety First**:  
  - Safe Mode (SELECT/EXPLAIN only)  
  - Single-statement guard  
  - Auto-LIMIT and statement timeout  
- **Schema Awareness**:  
  - Instant schema introspection  
  - Graphviz-powered ER diagrams  
- **Editable SQL**:  
  - Review, edit, and EXPLAIN before execution  
  - Download results as CSV  
- **Saved Queries**: Save, recall, and copy queries.
- **Modern UI**: Sleek glassmorphism cards, soft shadows, and a Simple Mode to hide advanced options.

---

## ⚡ Quick Start

### 1️⃣ Requirements
- Python 3.10+
- Reachable PostgreSQL instance
- System Graphviz (for schema diagrams)

### 2️⃣ Install Graphviz
- **Ubuntu/Debian:** `sudo apt-get install graphviz`
- **macOS:** `brew install graphviz`
- **Windows:** [Download & add to PATH](https://graphviz.gitlab.io/_pages/Download/Download_windows.html)

### 3️⃣ Setup Project
```bash
git clone https://github.com/Tariq3654467/Postgres_Agent.git
cd Postgres_Agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### `requirements.txt` includes:
- streamlit, sqlalchemy, psycopg2-binary, pandas, graphviz, crewai, python-dotenv

### 4️⃣ Configure Secrets
Create a `.env` file with:
```env
GROQ_API_KEY=gsk_your_real_groq_key_here
```
> Your key must start with `gsk_`. (OpenAI keys will not work.)

### 5️⃣ Run the App
```bash
streamlit run app.py
```
Open the browser at the shown URL.

---

## 💡 Usage

- Connect to PostgreSQL via the sidebar (host, port, db, user, password, GROQ_API_KEY).
- Click **Connect**.
- Chat with the agent!  
  **Example prompts:**
  - `list tables`
  - `top 50 orders by total`
  - `describe table customers`
  - `count users per country order by count desc`
- Review SQL before execution (edit as needed).
- Use **EXPLAIN** to preview the plan, or **Run** to execute.
- Browse schema and saved queries in the right panel.
- **Safety Tip:** Keep Safe Mode ON for read-only access.

---

## 🧠 Model Support

- Default: `groq/llama-3.1-8b-instant`
- Also supports:  
  `groq/llama3-8b-8192`,  
  `groq/llama3-70b-8192`,  
  `groq/mixtral-8x7b-32768`,  
  `groq/llama-3.3-70b-versatile`

Configure with `.env` or via the app sidebar.

---

## 🤝 Contributing

Pull requests welcome! For feature ideas or bug reports, open an issue.

---

## 📄 License

[MIT](LICENSE)

---

## 💬 Contact

Questions or feedback?  
Open an issue or contact [@Tariq3654467](https://github.com/Tariq3654467).
