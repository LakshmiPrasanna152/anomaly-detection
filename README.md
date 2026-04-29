Anomaly Detection in Network Traffic
Overview:

This project implements a robust anomaly detection pipeline designed to identify unusual patterns in network traffic data. It leverages multiple machine learning models and combines them using an ensemble voting mechanism to improve detection accuracy.
The system is capable of analyzing traffic behavior, detecting outliers, and generating insightful visualizations to support decision-making in cybersecurity and network monitoring.

 Key Features :

* 🔍 Multi-model anomaly detection (4 algorithms)
* 🧮 Ensemble voting (majority ≥ 2/4)
* 📊 Detailed performance metrics (Precision, Recall, F1-score)
* 🚨 Threat classification (DDoS, brute force, data exfiltration, etc.)
* 📈 Automated report generation with visualizations
* 🧪 Interactive CLI-based packet inspector
* 📦 Exportable datasets for further analysis


 Detection Pipeline :

Data Generation → Feature Engineering → Model Training  
→ Ensemble Voting → Threat Classification → Visualization → Interactive Analysis

 🤖 Models Implemented

Isolation Forest → High precision anomaly detection
Local Outlier Factor (LOF) → Density-based detection
DBSCAN → Cluster-based anomaly detection
Z-Score → Statistical anomaly detection


## 📊 Performance Summary

| Model            | Precision | Recall    | F1 Score  |
| ---------------- | --------- | --------- | --------- |
| Isolation Forest | 0.987     | 0.987     | 0.987     |
| LOF              | 0.262     | 0.262     | 0.262     |
| DBSCAN           | 0.860     | 1.000     | 0.925     |
| Z-Score          | 0.895     | 0.962     | 0.928     |
| **Ensemble**     | **0.860** | **1.000** | **0.925** |

📌 Ensemble improves robustness and recall across diverse anomaly types.



🚨 Threat Intelligence Breakdown

| Attack Type    | Description               |
| -------------- | ------------------------- |
| 🟡 Port Scan   | Reconnaissance activity   |
| 🔴 DDoS        | Service disruption attack |
| 💀 Data Exfil  | Critical data breach risk |
| 🔴 Brute Force | Credential attack         |
| 🟢 Normal      | Baseline traffic          |

## 🧪 Interactive Packet Inspector

The system includes a CLI tool where users can input packet features and instantly receive:

* ✅ Traffic classification (Normal / Anomaly)
* 🎯 Threat type prediction
* 📊 Anomaly score
* 🔄 Before vs after comparison

---

## 📂 Project Structure

```bash
anomaly-detection/
│
├── data/                      # Generated datasets
├── models/                    # Model artifacts
├── reports/
│   └── figures/               # Visual outputs
├── src/
│   └── detector.py            # Core pipeline
├── tests/                     # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
`
⚙️ Setup Instructions -


git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
 ▶️ Run the Project

python src/detector.py

 📊 Outputs Generated -

* 📈 Model comparison charts
* 📉 Anomaly distributions
* 🔄 Before vs After analysis plots
* 📁 Exported dataset: `data/traffic_with_anomalies.csv`


 📈 Key Insights -

* Ensemble detection achieves **100% recall**, ensuring no anomaly is missed
* Isolation Forest provides **highest precision (98.7%)**
* DBSCAN captures cluster-based anomalies effectively
* LOF underperforms → useful for comparison benchmarking

##  Real-World Applications-

* Cybersecurity intrusion detection systems
* Fraud detection pipelines
* Network monitoring tools
* SIEM (Security Information & Event Management) systems


 Future Enhancements

*  Deploy as a web dashboard (Streamlit / Flask)
*  Real-time streaming detection (Kafka / Spark)
*  Deep learning models (Autoencoders, LSTM)
* Integration with live network traffic



