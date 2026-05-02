<!-- 
  Stanbic Bank Ghana — SME Credit Assessment Pipeline
  Architecture Diagram (Mermaid format)
  
  To render: paste the code block below into https://mermaid.live
  Export as PNG at 2x resolution for the presentation deck.
-->

```mermaid
flowchart TD
    A["📋 Loan Officer Web Form\n(New SME Application)"]
    B["🔐 Azure API Management\n(Auth, Rate Limiting, Key Vault)"]
    C["⚡ Azure Functions\n(Validation & Routing)"]
    D["📄 Scanned Documents\n(Bank Statements, Registration Cert, TIN)"]
    E["🤖 LLM — Azure Document Intelligence\n(GPT-4 Vision: Extract structured fields\nfrom unstructured PDFs/images)"]
    F["🗄️ Azure Blob Storage\n(Raw document archive)"]
    G["⚙️ Preprocessing Module\nsrc/preprocessing.py\n(9 data quality fixes · Feature engineering)"]
    H["🧠 Azure ML Online Endpoint\nLogistic Regression Pipeline\nCV AUC = 0.61 · Gini = 0.22"]
    I["📊 Decision Engine\nsrc/decision_engine.py\nHard Rules R001–R005 · Threshold calibration"]
    J{"Recommendation?"}
    K["✅ APPROVE\n~30% of applications\nScore below threshold"]
    L["🔄 REFER\n~50% of applications\nUncertain zone"]
    M["❌ DECLINE\n~20% of applications\nScore above threshold\nor Hard Rule triggered"]
    N["🗃️ Azure SQL Database\n(Decision + SHAP explanation logged\nfor every application)"]
    O["👤 Relationship Manager\nReview Dashboard\n(Application + model score + SHAP)"]
    P["📝 RM Decision\n(APPROVE or DECLINE\nwith documented reason)"]
    Q["🔁 Retraining Pipeline\nAzure ML Pipelines\n(Triggers: AUC < 0.70 · PSI > 0.25\n· Every 6 months · 500+ new labels)"]
    R["📈 Azure Monitor\n+ Application Insights\n(Latency · Error rate · Decision drift)"]

    A --> B --> C
    D --> F --> E
    E --> G
    C --> G
    G --> H
    H --> I
    I --> J
    J --> K
    J --> L
    J --> M
    K --> N
    M --> N
    L --> O
    O --> P
    P --> N
    N --> Q
    R --> Q

    style A fill:#3498db,color:#fff,stroke:#2980b9
    style E fill:#9b59b6,color:#fff,stroke:#8e44ad
    style H fill:#2ecc71,color:#fff,stroke:#27ae60
    style I fill:#f39c12,color:#fff,stroke:#e67e22
    style K fill:#2ecc71,color:#fff,stroke:#27ae60
    style L fill:#f39c12,color:#fff,stroke:#e67e22
    style M fill:#e74c3c,color:#fff,stroke:#c0392b
    style O fill:#3498db,color:#fff,stroke:#2980b9
    style Q fill:#1abc9c,color:#fff,stroke:#16a085
```

## Key Design Decisions (for the diagram explanation)

| Component | Why it exists |
|-----------|--------------|
| LLM (Document Intelligence) | Extracts structured fields from scanned docs. Does NOT make credit decisions — it is a data extraction tool only. |
| Hard Rules (R001–R005) | Fire before the model. Categorical policy answers (e.g., 60+ DPD) that no probability changes. |
| Three-zone decision | No bank auto-decides every case. REFER zone explicitly handles uncertainty — humans review those. |
| SHAP on every DECLINE | Regulatory requirement. Every adverse decision must be explainable. |
| RM feedback → Retraining | Closes the loop. Human decisions become labeled training data for future model versions. |
| Azure ML Pipelines | Automated train → evaluate → register → deploy. Human approves before production swap. |

## Retraining Triggers

| Trigger | Threshold | Reason |
|---------|-----------|--------|
| Performance degradation | AUC on labeled batch < 0.70 | Model no longer discriminating |
| Data drift | PSI > 0.25 on key features | Input distribution shifted from training |
| Time-based | Every 6 months | Macroeconomic conditions change |
| Volume-based | 500+ new labeled samples | Enough fresh ground truth to retrain |
