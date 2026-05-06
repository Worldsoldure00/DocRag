# Multi-Agent RAG — Test Queries

A reference set of test questions organized by **expected routing domain**, complexity, and edge cases.  
Use these to verify that the router, retrieval, and synthesizer agents are all working correctly.

---

## 🏦 Finance Queries → `finance` domain

| # | Query |
|---|---|
| F1 | What was Apple's total revenue in fiscal year 2023? |
| F2 | How much net income did Microsoft report in 2023? |
| F3 | What are the key risk factors JPMorgan Chase identified in their 2023 10-K? |
| F4 | How much did Tesla spend on research and development in 2023? |
| F5 | What were Google's advertising revenues in 2022? |

---

## 🏥 Medical Queries → `medical` domain

| # | Query |
|---|---|
| M1 | What are the first-line treatment options for type 2 diabetes? |
| M2 | What is the mechanism of action of metformin? |
| M3 | How effective are PD-1 inhibitors in improving survival rates for lung cancer patients? |
| M4 | What lifestyle interventions are most effective for reducing cardiovascular disease risk? |
| M5 | What are the common side effects of antihypertensive medications? |

---

## 🔀 Cross-Domain Queries → `both` domain

Both the Finance and Medical agents are triggered, with the Synthesizer merging the results.

| # | Query |
|---|---|
| B1 | How did Pfizer's COVID-19 vaccine revenue affect their 2022 financials, and what are the clinical outcomes of their vaccine? |
| B2 | What is Johnson & Johnson's revenue from oncology drugs, and what does the research say about their cancer treatments? |
| B3 | How much does the pharmaceutical industry invest in diabetes drug R&D, and what new treatments are showing the most promise? |
| B4 | What are the financial projections for the cardiovascular drug market, and what are the most effective drugs being studied? |
| B5 | Explain the relationship between Pfizer's revenue growth and the clinical impact of their top-selling drugs. |

---

## 🌐 Web Fallback Queries

Topics outside the indexed documents — the system should fall back to the **Web Agent**.

| # | Query |
|---|---|
| W1 | What is the current stock price of Apple? |
| W2 | What are the latest FDA drug approvals in 2025? |
| W3 | Who is the current CEO of Goldman Sachs? |
| W4 | What are the side effects of a drug approved after 2024? |
| W5 | What happened at the most recent Federal Reserve meeting? |
