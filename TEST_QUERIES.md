# DocRag Test Queries

Use these queries to test the dynamic LangGraph routing and ensure the correct agent (Finance, Medical, Both, or Web Fallback) picks up the query.

## 📊 Finance Expert (SEC Filings)
These queries contain financial terminology and company names, triggering the `finance` routing path:
1. What was Apple's total net revenue for the fiscal year ended September 30, 2023?
2. What are the primary risk factors mentioned in Tesla's latest 10-K filing?
3. Summarize the total research and development (R&D) expenses reported in the latest quarter.

## 🏥 Medical Expert (PubMed Literature)
These queries contain clinical terminology, triggering the `medical` routing path:
1. What is the relationship between apical vertebra rotation and severe scoliosis?
2. What are the recommended clinical treatments for supine hypertension?
3. Does continuous positive airway pressure (CPAP) treatment reduce blood pressure in patients with obstructive sleep apnea?

## ⚖️ Both Experts (Hybrid Intersectional Queries)
These queries contain a mix of financial terms (cost, revenue, market) and medical terms (hypertension, clinical), triggering the `both` routing path:
1. What is the financial cost and market impact associated with clinical treatments for severe hypertension?
2. Are there any financial reports on the revenue growth of healthcare companies developing treatments for scoliosis?
3. How do medical technology investments and R&D spending impact the treatment outcomes for cardiovascular diseases?

## 🌐 Web Fallback Expert (Out-of-Domain)
These queries have absolutely nothing to do with SEC filings or PubMed literature. The router will likely guess `medical` or `both`, but the local vector databases will return `0` sources, causing the graph to dynamically fallback to the `web_agent`:
1. What is the capital of France and what is its population?
2. How do I bake a homemade chocolate cake from scratch?
3. Who won the Super Bowl in 2024?
