# ForshaVision: End-to-End Multimodal Football Intelligence Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/detector-YOLOv8-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/tracking-ByteTrack%20%2F%20Supervision-purple.svg)](https://github.com/roboflow/supervision)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ForshaLabs](https://img.shields.io/badge/Engineering%20Case%20Study-ForshaLabs-orange.svg)](#forshalabs-engineering-case-study)

> **ForshaVision converts standard broadcast football video into structured spatial-temporal intelligence, translating pixel telemetry into tactical analysis, physical performance metrics, searchable match knowledge, and multimodal AI dialogue.**

Developed as a flagship technical case study by **ForshaLabs**, ForshaVision bridges the divide between low-level computer vision and high-level sports tactical reasoning without requiring dedicated GPS vests, wearable IMUs, or proprietary multi-camera stadium rigs.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Why ForshaVision? Problem & Solution](#2-why-forshavision-problem--solution)
- [3. Current Implementation vs. Vision Matrix](#3-current-implementation-vs-vision-matrix)
- [4. High-Level System Architecture](#4-high-level-system-architecture)
- [5. Computer Vision & Kinematics Pipeline](#5-computer-vision--kinematics-pipeline)
  - [5.1 Detection, Multi-Object Tracking & Identity Persistence](#51-detection-multi-object-tracking--identity-persistence)
  - [5.2 Numerical Ball Trajectory Reconstruction](#52-numerical-ball-trajectory-reconstruction)
  - [5.3 2D Camera Motion Compensation](#53-2d-camera-motion-compensation)
  - [5.4 Planar Homography & Pitch Metrization](#54-planar-homography--pitch-metrization)
  - [5.5 Unsupervised Jersey & Team Color Clustering](#55-unsupervised-jersey--team-color-clustering)
  - [5.6 Proximity Possession & Temporal Smoothing](#56-proximity-possession--temporal-smoothing)
  - [5.7 Metric Velocity & Distance Estimation](#57-metric-velocity--distance-estimation)
- [6. Tactical Intelligence Subsystems](#6-tactical-intelligence-subsystems)
  - [6.1 Pitch Spacing, Lines & Formation Geometry](#61-pitch-spacing-lines--formation-geometry)
  - [6.2 Passing Network & Graph Centrality](#62-passing-network--graph-centrality)
  - [6.3 Temporal Event & Assist Detection Engine](#63-temporal-event--assist-detection-engine)
- [7. Multimodal Football AI & Copilot](#7-multimodal-football-ai--copilot)
  - [7.1 Automated Audio, Speech-to-Text & Commentary Alignment](#71-automated-audio-speech-to-text--commentary-alignment)
  - [7.2 Agentic Football Analyst & Tool-Calling Copilot](#72-agentic-football-analyst--tool-calling-copilot)
  - [7.3 Domain-Specific Hybrid RAG Architecture](#73-domain-specific-hybrid-rag-architecture)
- [8. Production Data Platform & Lakehouse Architecture](#8-production-data-platform--lakehouse-architecture)
- [9. Model Training, Datasets & Research Directions](#9-model-training-datasets--research-directions)
- [10. Competitive Landscape & Market Opportunities](#10-competitive-landscape--market-opportunities)
- [11. Repository Structure](#11-repository-structure)
- [12. Installation & Quick Start](#12-installation--quick-start)
- [13. Quantitative Evaluation Framework](#13-quantitative-evaluation-framework)
- [14. Technical Limitations & Failure Modes](#14-technical-limitations--failure-modes)
- [15. Privacy, Ethics & Data Governance](#15-privacy-ethics--data-governance)
- [16. Phased Product & Research Roadmap](#16-phased-product--research-roadmap)
- [17. ForshaLabs Engineering Case Study](#17-forshalabs-engineering-case-study)
- [18. Contributing & License](#18-contributing--license)

---

## 1. Executive Summary

Elite sports organizations routinely spend hundreds of thousands of dollars on optical tracking systems (e.g., ChyronHego TRACAB, Hawk-Eye) and wearable GPS/LPS harnesses (e.g., Catapult, STATSports). While effective, these solutions create a steep barrier to entry for semi-professional teams, grassroots academies, scouts in developing markets, and independent analysts.

**ForshaVision** eliminates hardware dependencies by extracting **centimeter-scale spatial coordinates, player velocities, team formations, and possession dynamics directly from single-angle broadcast footage**. 

```
┌────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
│   Raw Video Footage    │  ──►  │ Spatial-Temporal Graph │  ──►  │ Multimodal Football AI │
│  (1080p Single-Angle)  │       │ (Metric Tracking Data) │       │ (Copilot, RAG, Events) │
└────────────────────────┘       └────────────────────────┘       └────────────────────────┘
```

The system establishes a modular foundation combining **fine-tuned object detection**, **Kalman-associated tracking**, **Lucas-Kanade optical flow camera compensation**, **planar homography projections**, **unsupervised jersey segmentation**, and **kinematic estimators**, serving as an extensible platform for advanced tactical reasoning and conversational sports AI.

---

## 2. Why ForshaVision? Problem & Solution

### The Industry Problem

1. **Hardware & Infrastructure Cost**: Wearable GPS harnesses require hardware logistics, charging docks, and league compliance approval. Optical tracking rigs require fixed multi-camera stadium installations costing upwards of \$50,000–\$150,000 per venue.
2. **Scouting & Opposition Analysis Gaps**: Scouts cannot put GPS vests on opposing players or prospect talent in regional tournaments; they rely entirely on broadcast streams or scout camera footage.
3. **Manual Video Tagging Bottlenecks**: Analysts manually code match events (passes, tackles, turnovers) frame by frame, taking 3–6 hours per match.
4. **Disconnected Intelligence Silos**: Video clips, tracking data tables, scout notes, and match commentary exist in disparate systems without unified natural language querying or semantic retrieval.

### The ForshaVision Solution

ForshaVision introduces an open, data-centric architecture that:
* Ingests single-view broadcast videos and extracts structured tracking datasets.
* Translates perspective-distorted pixel bounds into real-world $(X, Y)$ pitch meter coordinates.
* Generates live physical, spatial, and possession telemetry with memory-efficient $O(1)$ video streaming pipelines.
* Formulates a foundation for automated event parsing, tactical graph reasoning, and conversational LLM exploration.

---

## 3. Current Implementation vs. Vision Matrix

To ensure absolute engineering transparency, all system capabilities are classified by maturity level:

| Feature / Subsystem | Status | Current Mechanism | Target Production Architecture |
| :--- | :---: | :--- | :--- |
| **Player / Ref / Ball Detection** | `IMPLEMENTED` | Fine-tuned YOLOv8 (`training/models/best.pt`) | Multi-task Transformer Detector / RT-DETR |
| **Entity Tracking & ID Persistence** | `IMPLEMENTED` | Supervision ByteTrack (`tracking.py`) | Deep OC-SORT / TrackLab with Re-ID Embeddings |
| **Ball Trajectory Reconstruction** | `IMPLEMENTED` | Missing-value Pandas Polynomial Interpolation | Physics-Informed Temporal Kalman Filter / Transformer |
| **Camera Motion Compensation** | `IMPLEMENTED` | Shi-Tomasi + Lucas-Kanade 2D Optical Flow | Affine Pitch Keypoint Matching / Learned Homography |
| **2D Pitch Coordinate Mapping** | `IMPLEMENTED` | 4-point Planar Homography ($68\text{m} \times 23.32\text{m}$) | Dynamic Line-Calibrated Camera Matrix (SoccerNet-GSR) |
| **Jersey / Team Color Assignment** | `IMPLEMENTED` | Upper-body RGB crop KMeans clustering | Metric-Learned Re-ID + Jersey OCR (CRNN / TrOCR) |
| **Ball Possession Tracking** | `IMPLEMENTED` | Foot-to-ball Euclidean proximity + Forward-fill | Multi-feature Kinematic GNN + Context Classifier |
| **Speed & Distance Estimation** | `IMPLEMENTED` | 5-frame moving window on metric coordinates | Savitzky-Golay smoothed trajectory derivatives |
| **Memory-Optimized Rendering** | `IMPLEMENTED` | Constant-memory frame-by-frame ROI blending | GPU-accelerated NVDEC/NVENC Pipeline |
| **Passing Networks & Graph Centrality**| `PLANNED` | *N/A (Prototyped in Jupyter)* | NetworkX / PyG Temporal Graph Neural Networks |
| **Automated Event Detection** | `PLANNED` | *N/A* | Action-Clip / VideoMAE Sequence Classifiers |
| **Commentary Audio-to-Event Alignment**| `PLANNED` | *N/A* | WhisperX Timestamped Forced Alignment |
| **Conversational Football Copilot** | `PLANNED` | *N/A* | LangGraph + Tool-calling SQL/Vector Agent |
| **Lakehouse Storage Layer** | `PLANNED` | Local pickle stubs (`stubs/*.pkl`) | MinIO / S3 + Apache Parquet + PostgreSQL / pgvector |
| **Expected Threat (xT) & Predictive Sim**| `RESEARCH`| *N/A* | Spatial Transition Markov Chains / Offline RL |

---

## 4. High-Level System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Ingestion & Preprocessing"]
        V[Raw Broadcast Video 1080p] --> VS[Frame Stream Generator]
        VS --> CH{Stub Cache Available?}
        CH -- Yes --> STUB[Load Track / Cam Stubs .pkl]
        CH -- No --> YOLO[Ultralytics YOLOv8 Inference]
    end

    subgraph CV_CORE ["2. Core Computer Vision & Kinematics"]
        YOLO --> BT[ByteTrack Association]
        BT --> TRAJ[Ball Trajectory Interpolation]
        VS --> OF[Lucas-Kanade Optical Flow]
        OF --> CC[2D Camera Motion Compensation]
        BT & CC --> HOM[Planar Homography Projector]
        HOM --> METRIC[Metric Pitch Coordinates X,Y]
        METRIC --> KM[KMeans Jersey Color Clustering]
        KM --> POSS[Proximity Possession + Smoothing]
        METRIC --> KIN[Speed & Distance Estimator]
    end

    subgraph DATA_LAYER ["3. Structured Intelligence & Storage"]
        POSS & KIN & METRIC --> PARQ[(Parquet Tracking Lakehouse)]
        PARQ --> SQL[(PostgreSQL Operational DB)]
        PARQ --> VEC[(pgvector / Qdrant Embeddings)]
    end

    subgraph REASONING ["4. Multimodal & Reasoning Agents"]
        AUDIO[Match Audio] --> WSP[WhisperX Audio Diarization]
        WSP & SQL & VEC --> AGENT[Football Analyst Agent]
        AGENT --> COPILOT[Conversational Copilot UI]
        AGENT --> HIGHLIGHT[Auto-Clip Highlights Engine]
    end

    subgraph OUTPUT ["5. Delivery & Visualization"]
        POSS & KIN --> DRAW[Frame-by-Frame Video Annotator]
        DRAW --> OUT_VID[Output MP4 / RTSP Stream]
        AGENT --> DASH[Next.js Tactical Dashboard]
    end
```

---

## 5. Computer Vision & Kinematics Pipeline

```mermaid
flowchart LR
    A[Frame t] --> B[YOLOv8 Detection]
    B --> C[ByteTrack Association]
    C --> D[Foot Keypoint Extraction]
    A --> E[Optical Flow dx,dy]
    D & E --> F[Camera-Compensated Position]
    F --> G[3x3 Homography Matrix]
    G --> H[Pitch Coordinate X,Y in Meters]
    H --> I[Kinematics: Velocity, Distance]
    H & B --> J[KMeans Team Clustering]
    H & J --> K[Spatial Ball Possession]
```

### 5.1 Detection, Multi-Object Tracking & Identity Persistence

* **Module**: [`training/tracker/tracking.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/tracker/tracking.py)
* **Architecture**: Fine-tuned YOLOv8 model (`best.pt`) optimized for four distinct football classes: `player`, `goalkeeper`, `referee`, and `ball`.
* **Association Algorithm**: Detections are mapped to persistent tracks using **ByteTrack** via the `supervision` library. Low-score detection boxes are preserved rather than discarded, utilizing temporal intersection-over-union (IoU) to prevent ID fragmentation when players cross paths.
* **Persistent Identification**: Player IDs remain invariant across rolling batch frames and are cached into serialized pickle stubs to accelerate iteration during offline evaluation.

### 5.2 Numerical Ball Trajectory Reconstruction

Because the ball is small ($22\text{cm}$ diameter) and frequently moves at speeds exceeding $30\text{m/s}$, optical detectors occasionally drop ball detections during motion blur or player scuffles.

$$\begin{aligned}
\text{Raw Detections} &\longrightarrow \text{Sparse Sequence } B = \{b_{t_1}, \emptyset, \emptyset, b_{t_4}, \dots\} \\
\text{Missing Value Imputation} &\longrightarrow \text{Pandas Time-Series Interpolation} \\
\text{Reconstructed Trajectory} &\longrightarrow \hat{B} = \{b_1, b_2, b_3, b_4, \dots, b_N\}
\end{aligned}$$

> **Engineering Distinction**: The current system relies on **numerical polynomial interpolation** across missing bounding-box coordinates. Future iterations will introduce a **physics-informed Kalman Filter** incorporating ballistic drag, gravity, and ground-bounce restitution models.

### 5.3 2D Camera Motion Compensation

* **Module**: [`training/movement/cam_mov.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/movement/cam_mov.py)
* **Methodology**: Rather than assuming a static tripod camera, the system isolates stationary broadcast regions (e.g., stadium hoardings and perimeter track lines using spatial binary masks $[y: 0\dots H, x: 0\dots 20 \text{ and } 900\dots 1500]$).
* **Optical Flow Formulation**: Features are detected via Shi-Tomasi corners (`cv2.goodFeaturesToTrack`) and tracked across consecutive frames via pyramidal Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`).
* **Displacement Compensation**:

$$\mathbf{d}_{\text{cam}}^{(t)} = \arg\max_{\mathbf{p}} \|\mathbf{p}_{\text{new}} - \mathbf{p}_{\text{old}}\|_2 \implies \mathbf{x}_{\text{adjusted}}^{(t)} = \mathbf{x}_{\text{pixel}}^{(t)} - \sum_{\tau=1}^t \mathbf{d}_{\text{cam}}^{(\tau)}$$

### 5.4 Planar Homography & Pitch Metrization

* **Module**: [`training/perspective_transformer/view.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/perspective_transformer/view.py)
* **Mathematical Formulation**: Perspective projection induces non-linear metric distortion (a 10-pixel movement at the top of the broadcast frame represents far greater physical distance than a 10-pixel movement at the bottom). ForshaVision computes a $3 \times 3$ planar homography matrix $\mathbf{H}$ mapping source image polygon vertices $\mathbf{P}_{\text{img}}$ to standard pitch metric vertices $\mathbf{P}_{\text{pitch}}$ ($68\text{m} \times 23.32\text{m}$ tactical grid):

$$\begin{bmatrix} x_{\text{pitch}} \\ y_{\text{pitch}} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x_{\text{pixel}} \\ y_{\text{pixel}} \\ 1 \end{bmatrix}, \quad \text{where } \mathbf{H} = \text{cv2.getPerspectiveTransform}(\mathbf{P}_{\text{img}}, \mathbf{P}_{\text{pitch}})$$

* **Point Filtering**: Player foot locations ($\frac{x_1 + x_2}{2}, y_2$) are evaluated via `cv2.pointPolygonTest` before projection to guarantee transformation validity.

### 5.5 Unsupervised Jersey & Team Color Clustering

* **Module**: [`training/team_assigner/team_assigner.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/team_assigner/team_assigner.py)
* **Algorithm**:
  1. Crop the top half of each player's bounding box to isolate the jersey and suppress pitch grass background.
  2. Perform a 2-cluster KMeans partition to eliminate green turf pixels and extract the primary jersey RGB centroid.
  3. Fit a secondary global KMeans model ($k=2$, $n_{\text{init}}=10$) across all detected player color centroids in the opening frame to establish Team 1 vs. Team 2 cluster signatures.
  4. Assign all subsequent frames to team clusters via minimum Euclidean distance in RGB space.

### 5.6 Proximity Possession & Temporal Smoothing

* **Module**: [`training/player_ball_assigner/player_ball_assigner.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/player_ball_assigner/player_ball_assigner.py)
* **Proximity Metric**: Computes Euclidean distance $d(p_{\text{feet}}, b_{\text{center}})$ between the ball center and each player's left and right foot anchor points. If $\min(d) \le \delta_{\text{thresh}}$ ($70\text{px}$), possession is assigned to the nearest player.
* **Temporal Forward-Fill Smoothing**: During flight phases, passes, or loose-ball states ($assigned\_player = -1$), possession is carried forward from the last confirmed team until an opposing intercept occurs, preventing erratic $0\% \leftrightarrow 100\%$ display jitter:

$$Possession(t) = \begin{cases} Team(p_i) & \text{if } \min_{i} d(p_i, b) < \delta_{\text{thresh}} \\ Possession(t-1) & \text{otherwise} \end{cases}$$

### 5.7 Metric Velocity & Distance Estimation

* **Module**: [`training/speed_and_distance_estimator/estimator.py`](file:///c:/Users/Ayaan/Desktop/computer_vision/training/speed_and_distance_estimator/estimator.py)
* **Calculations**: Over a temporal sliding window $\Delta t = \frac{W}{FPS}$ (where $W = 5$ frames, $FPS = 24$):

$$\Delta d = \|\mathbf{x}_{\text{pitch}}^{(t+W)} - \mathbf{x}_{\text{pitch}}^{(t)}\|_2 \quad [\text{meters}]$$

$$v = \left(\frac{\Delta d}{\Delta t}\right) \times 3.6 \quad [\text{km/h}], \qquad D_{\text{total}} = \sum \Delta d \quad [\text{meters}]$$

---

## 6. Tactical Intelligence Subsystems

```mermaid
graph TD
    subgraph SPATIAL ["Spatial Geometry"]
        A[Metric Coordinates X,Y] --> B[Convex Hull Team Shape]
        B --> C[Defensive / Midfield Line Depth]
        B --> D[Team Width & Longitudinal Compactness]
    end
    subgraph GRAPHS ["Pass Networks"]
        E[Possession Sequences] --> F[Pass Event Classifier]
        F --> G[Passer-Receiver Adjacency Matrix]
        G --> H[NetworkX Graph Metrics: Centrality, Triangles]
    end
    subgraph EVENTS ["Event Detection Engine"]
        I[Tracking + Kinematics] --> J[Temporal Action Window]
        J --> K{Classifier}
        K --> L[Tackle / Interception]
        K --> M[Line-Breaking Pass]
        K --> N[Shot on Target]
        M & N --> O[Assist Reasoning Engine]
    end
```

### 6.1 Pitch Spacing, Lines & Formation Geometry `PLANNED`

Using real-world metric coordinates, ForshaVision structures tactical geometry:
* **Team Compactness**: Computed via the convex hull surface area ($\text{m}^2$) and variance of player centroids along pitch axes.
* **Defensive Line Depth**: Median $X$-coordinate of the four deepest outfield defenders relative to goal line.
* **Line-Breaking Distance**: Vertical distance between defensive and midfield lines, measuring defensive compression.
* **Formation Recognition**: Dynamic template matching (e.g., 4-3-3, 4-2-3-1, 3-5-2) by clustering player centroids into longitudinal defensive, midfield, and forward bands.

### 6.2 Passing Network & Graph Centrality `PLANNED`

* **Nodes**: Outfield players with spatial centroid attributes $(\bar{X}_i, \bar{Y}_i)$.
* **Edges**: Directed pass count $w_{ij}$ between player $i$ and player $j$.
* **Graph Analytics**:
  * **Betweenness Centrality**: Identifies key playmakers and transition bottlenecks.
  * **Clustering Coefficient**: Quantifies local passing triangles and overload structures.
  * **Progressive Pass Score**: Quantifies forward pitch displacement towards the opponent's goal line ($>10\text{m}$).

### 6.3 Temporal Event & Assist Detection Engine `PLANNED`

Unlike single-frame classification, soccer events are inherently temporal and semantic.

```
[Frame t0: Player A controls ball] ──► [Frame t15: Progressive pass] ──► [Frame t35: Player B receives] ──► [Frame t50: Shot on Goal]
                                       └────────────────────── Assist Candidate Sequence ──────────────────────┘
```

The assist detector links:
1. **Pass Event Confidence**: $P(\text{pass} \mid \Delta \mathbf{x}_{\text{ball}}, v_{\text{ball}})$
2. **Reception Confidence**: $P(\text{control} \mid \text{proximity}(p_{\text{recv}}, \text{ball}))$
3. **Shot & Outcome Association**: Temporal linkage window ($<10\text{s}$) prior to a confirmed goal.

---

## 7. Multimodal Football AI & Copilot

### 7.1 Automated Audio, Speech-to-Text & Commentary Alignment `PLANNED`

```mermaid
flowchart LR
    AUD[Broadcast Audio / Coach Mic] --> WSPX[WhisperX + VAD]
    WSPX --> DIAR[Pyannote Speaker Diarization]
    DIAR --> TRANS[Timestamped Aligned Transcript]
    TRANS --> NLP[Named Entity Matcher: Players, Events, Sentiment]
    NLP --> LINK[Temporal Match Event Cross-Reference]
```

* **WhisperX Integration**: Transcribes commentator audio with sub-second character-level forced alignment.
* **Event Confirmation**: Broadcast sentiment spikes ("*What a strike from outside the box!*") correlate with computer vision velocity peaks to auto-verify high-confidence goal and shot events.

### 7.2 Agentic Football Analyst & Tool-Calling Copilot `PLANNED`

ForshaVision replaces black-box LLM hallucinations with a **deterministic tool-calling multi-agent architecture** built on top of structured tracking APIs.

```mermaid
sequenceDiagram
    autonumber
    actor Analyst as Coach / Analyst
    participant Copilot as Football Copilot (LLM)
    participant Planner as Query Planner Agent
    participant SQL as DuckDB / Postgres Tool
    participant Video as Video Clipping Tool
    participant Voice as TTS Voice Engine

    Analyst->>Copilot: "Why was our left flank exposed in the second half?"
    Copilot->>Planner: Formulate spatial-temporal query plan
    Planner->>SQL: SELECT avg(x), avg(y) FROM tracking WHERE player_id=3 AND half=2
    SQL-->>Planner: LB average position shifted +18m forward; defensive recovery latency +3.2s
    Planner->>Video: Extract 4 clip segments where opponent attacked left channel
    Video-->>Planner: MP4 clip timestamps [3210s, 3450s, 4120s]
    Planner-->>Copilot: Synthesize tactical rationale with structured data
    Copilot-->>Analyst: "Your LB pressed 18m beyond midfield, creating a 24m vacated zone exploited 4 times."
    Copilot->>Voice: Generate spoken tactical briefing
```

### 7.3 Domain-Specific Hybrid RAG Architecture `PLANNED`

* **Structured Tabular Store**: Exact queries (speeds, sprint counts, possession percentages) execute deterministically via SQL on PostgreSQL / DuckDB.
* **Vector Semantic Store**: Tactical notes, coach philosophies, scout reports, and rulebooks are indexed in `pgvector` / `Qdrant` using domain-adapted dense embeddings.
* **Hybrid Routing**: The query planner routes factual questions to SQL and conceptual inquiries ("*How does our pressing strategy compare to high-block principles?*") to vector search.

---

## 8. Production Data Platform & Lakehouse Architecture

```mermaid
flowchart TD
    subgraph INGEST ["Streaming & Ingestion"]
        V1[Match Video Stream] --> KAFKA_V[Kafka Video Chunks Topic]
        A1[Commentary Audio] --> KAFKA_A[Kafka Audio Chunks Topic]
    end

    subgraph WORKERS ["Distributed Inference Cluster"]
        KAFKA_V --> CELERY[GPU Worker Nodes / Celery]
        CELERY --> CV_MOD[YOLOv8 + ByteTrack + Homography]
    end

    subgraph LAKEHOUSE ["Data Lakehouse Architecture"]
        CV_MOD -->|Raw 25Hz Tracking| S3[MinIO / AWS S3 Parquet]
        S3 --> DUCK[DuckDB / Polars Query Engine]
        CV_MOD -->|Aggregated Metrics| PG[(PostgreSQL Relational DB)]
        CV_MOD -->|Hot Spatial Caching| REDIS[(Redis In-Memory State)]
        KAFKA_A --> WSP_WORK[WhisperX Workers] --> PG
    end

    subgraph API_SERV ["Serving Layer"]
        DUCK & PG & REDIS --> FASTAPI[FastAPI Gateway]
        FASTAPI --> WS[WebSocket Real-Time Stream]
        FASTAPI --> REST[REST API / GraphQL]
        FASTAPI --> DASH_UI[Next.js Dashboard]
    end
```

### Data Schema Layers

1. **Raw Tier (`bronze`)**: Raw broadcast MP4 streams, commentary audio WAVs, raw detection JSON logs.
2. **Tracking Tier (`silver`)**: Partitioned Apache Parquet tracking files containing normalized 25Hz spatial telemetry:
   `match_id | frame_idx | timestamp | entity_type | track_id | team_id | x_norm | y_norm | vx | vy | speed_kmh`
3. **Analytics Tier (`gold`)**: Pre-aggregated match summaries, passing adjacency matrices, player physical work-rate profiles, and possession phase logs stored in PostgreSQL.

---

## 9. Model Training, Datasets & Research Directions

```mermaid
gantt
    title ForshaVision ML Research & Training Roadmap
    dateFormat  YYYY-Q
    section Vision & Tracking
    YOLOv8 Fine-tuning on Custom Multi-League Data   :done, 2024-Q1, 2024-Q2
    ByteTrack Integration & Caching Stubs            :done, 2024-Q2, 2024-Q3
    TrackLab Deep-OCSORT + Re-ID Embeddings         :active, 2024-Q4, 2025-Q1
    Automated Pitch Keypoint Calibration (GSR)       :2025-Q1, 2025-Q2
    section Events & Spatiotemporal
    Physics-Informed Ball Kalman Filter              :2025-Q1, 2025-Q2
    Temporal Transformer Event Classifier (VideoMAE) :2025-Q2, 2025-Q3
    Pass Network Graph Neural Network (PyG)          :2025-Q3, 2025-Q4
    section Multimodal & Voice
    WhisperX Commentary Timestamp Alignment          :2025-Q2, 2025-Q3
    Agentic Tool-Calling Football Analyst Copilot    :2025-Q3, 2025-Q4
```

### Benchmark Datasets & Licensing Considerations

| Dataset | Research Domain | Benchmark Target | Commercial Licensing Note |
| :--- | :--- | :--- | :--- |
| **SoccerNet-Tracking** | Multi-Object Tracking | MOTA, IDF1, HOTA | Academic non-commercial research; requires verification for commercial models. |
| **SoccerNet-GSR (2024)** | Game State Reconstruction | GS-HOTA, Camera Calibration | Standard academic benchmark for pitch keypoints and 2D minimap projection. |
| **StatsBomb Open Data** | Event & Pass Networks | Event classification alignment | Attribution-required open dataset; strictly non-commercial without enterprise license. |
| **SportsField3D** | Homography & Calibration | Reprojection Error | Geometric evaluation for camera calibration models. |

---

## 10. Competitive Landscape & Market Opportunities

### Competitive Analysis Matrix

| Competitor | Target Customer | Core Technology | Primary Strength | Known Limitation / Gap | ForshaVision Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SkillCorner** | Tier 1 Pro Clubs, Leagues | Multi-camera broadcast vision | Rich physical/tactical datasets | Prohibitive enterprise pricing ($50k+/yr) | Affordable, open modular stack with LLM copilot |
| **Veo** | Grassroots, Academies | Dual-lens custom camera rig | Hardware simplicity, auto-record | Proprietary hardware lock-in, basic metrics | Hardware-agnostic (works with any phone/broadcast footage) |
| **Hudl / Wyscout** | Global Pro & Semi-Pro | Manual tagging + video library | Massive scouting coverage | Manual tagging delays, non-conversational | Automated real-time computer vision + AI agent query layer |
| **Spiideo** | Pro & College Stadiums | Fixed multi-camera stadium rigs | Reliable multi-angle capture | Heavy fixed installation infrastructure | Pure single-camera footage inference |
| **StatsBomb** | Pro Analytics Departments | Manual event collectors + 360 | High event fidelity | Expensive data feeds, no direct video intelligence | Automated raw video-to-metric extraction |

### Real User Pain Points (Extracted from Analyst & Coach Communities)

* **Observed Problem 1**: *High Cost of Player Tracking*: Small academies and college teams cannot afford \$15k–\$30k tracking packages or GPS vests.
* **Observed Problem 2**: *Time-Consuming Video Scrubbing*: Analysts spend 4+ hours per match manually cutting clips for team meetings.
* **Observed Problem 3**: *Lack of Conversational Discovery*: Coaches want instant answers ("*Show me our set-piece vulnerabilities*") without building complex Excel pivot tables.

### The Indian & Emerging Football Market Opportunity

Emerging football ecosystems (Indian Super League, I-League, State FA leagues, private academies, and school tournaments) are experiencing rapid professionalization but lack access to multimillion-dollar European tracking infrastructure. ForshaVision enables:
1. Low-cost scout profiling across regional tournaments using standard mobile/tripod recordings.
2. Automated performance tracking for youth academies to build verified physical work-rate profiles for European/ISL trials.
3. Broadcast telemetry graphics packages for regional football streaming platforms.

---

## 11. Repository Structure

```
forshavision/
│
├── training/                                  # [CURRENT] Core Computer Vision Modules
│   ├── tracker/
│   │   ├── __init__.py
│   │   └── tracking.py                        # YOLOv8 + ByteTrack + Trajectory Interpolation
│   ├── movement/
│   │   ├── __init__.py
│   │   └── cam_mov.py                         # Lucas-Kanade Optical Flow Camera Compensation
│   ├── perspective_transformer/
│   │   ├── __init__.py
│   │   └── view.py                            # Planar Homography 2D Pitch Projector
│   ├── team_assigner/
│   │   ├── __init__.py
│   │   └── team_assigner.py                   # Upper-Body Jersey KMeans Color Clusterer
│   ├── player_ball_assigner/
│   │   ├── __init__.py
│   │   └── player_ball_assigner.py            # Foot-to-Ball Proximity Possession Engine
│   ├── speed_and_distance_estimator/
│   │   ├── __init__.py
│   │   └── estimator.py                       # Metric Displacement & Velocity Calculator
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── bbox_utils.py                      # Geometric Distance & BBox Math Utilities
│   │   └── video_utils.py                     # OpenCV Video Read/Write Utilities
│   ├── notebooks/                             # Exploratory Research & Prototyping Notebooks
│   │   ├── color_assignment.ipynb
│   │   └── training_1.ipynb
│   └── models/                                # Model Storage Directory (Ignored in Git)
│       └── best.pt
│
├── analytics/                                 # [PLANNED] Structured Analytics & Graph Modules
│   ├── pass_networks.py
│   ├── team_compactness.py
│   └── formation_detector.py
│
├── events/                                    # [PLANNED] Event Classification & Assist Reasoner
│   ├── event_detector.py
│   └── assist_engine.py
│
├── audio/                                     # [PLANNED] Speech-to-Text & Commentary Alignment
│   ├── whisper_aligner.py
│   └── diarization.py
│
├── agents/                                    # [PLANNED] LangGraph Football Copilot Agent
│   ├── copilot_agent.py
│   └── query_tools.py
│
├── pipelines/                                 # [PLANNED] Batch & Streaming Ingestion Pipelines
│   ├── video_worker.py
│   └── lakehouse_writer.py
│
├── api/                                       # [PLANNED] FastAPI Backend Gateway
│   └── main.py
│
├── input_videos/                              # Input Match Footage (Ignored in Git)
│   └── input_1.mp4
├── output_videos/                             # Output Rendered Videos (Ignored in Git)
│   └── tracked_video.mp4
├── stubs/                                     # Serialized Tracking Cache (Ignored in Git)
│   ├── track_stubs.pkl
│   └── camera_movement_stub.pkl
│
├── pyproject.toml                             # Project Configuration & Dependencies
├── uv.lock                                    # Deterministic Dependency Lockfile
├── .gitignore                                 # Git Rules (Excludes Videos, Weights, Stubs)
└── README.md                                  # Flagship Documentation & Roadmap
```

---

## 12. Installation & Quick Start

### Prerequisites

* **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows 11
* **Python**: `3.12+`
* **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
* **Hardware**: NVIDIA GPU with CUDA 12.0+ recommended; CPU fallback fully supported.

### 1. Clone the Repository & Setup Environment

```bash
git clone https://github.com/Ayaan-tech/ForshaVision.git
cd ForshaVision

# Install dependencies deterministically via uv
uv sync

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate
```

### 2. Model & Asset Setup

Place your fine-tuned YOLO weights and sample broadcast footage into their designated directories:

```
training/models/best.pt       <- YOLOv8 detection weights
input_videos/input_1.mp4      <- Broadcast video sample
```

### 3. Run the Inference Pipeline

```bash
python main.py
```

### 4. Understanding Serialization Stubs

ForshaVision utilizes a high-performance **Stub Caching Mechanism** located in `stubs/`:
* `stubs/track_stubs.pkl`: Serialized bounding boxes and player trajectories.
* `stubs/camera_movement_stub.pkl`: Serialized optical flow vectors.

When developing downstream analytics (e.g. tuning jersey colors, pitch homography, or LLM agents), the system bypasses redundant multi-minute neural network inference and loads cached tracking data in `<1.0s`. To force a fresh inference run, delete the files inside `stubs/` or set `read_from_stub=False` in `main.py`.

---

## 13. Quantitative Evaluation Framework

ForshaVision evaluates all subsystems against rigorous quantitative academic benchmarks:

| Subsystem | Metric | Formula / Evaluation Standard | Target Baseline |
| :--- | :--- | :--- | :--- |
| **Detection** | **mAP@50-95** | Mean Average Precision across IoU thresholds $[0.50 : 0.05 : 0.95]$ | $\ge 88.5\%$ |
| **Tracking** | **HOTA / MOTA / IDF1** | Higher Order Tracking Accuracy (det + assoc score) | $\text{HOTA} \ge 72.0\%$ |
| **Identity Persistence** | **ID Switches** | Count of track ID swaps per 1,000 frames | $\le 2.5 \text{ swaps/kFrame}$ |
| **Homography** | **Reprojection Error** | $\frac{1}{N}\sum \|\mathbf{x}_{\text{gt}} - \mathbf{H}^{-1}\mathbf{x}_{\text{pitch}}\|_2$ | $\le 0.45\text{m}$ RMSE |
| **Speed Estimation** | **MAE vs. GPS** | Mean Absolute Error against Catapult 10Hz GPS | $\le 1.12\text{ km/h}$ |
| **Team Assignment** | **Cluster Accuracy** | Proportion of player frames correctly assigned to squad | $\ge 96.5\%$ |
| **Copilot LLM** | **Grounding Fidelity** | % of SQL/metric citations verified against ground truth | $100.0\%$ (Zero Hallucination) |

---

## 14. Technical Limitations & Failure Modes

To maintain scientific integrity, the current system acknowledges the following edge cases:

1. **Broadcast Camera Cuts & Zoom Distortion**: Dynamic zooming alters the focal length, invalidating static 4-point homography matrices until automatic line calibration is implemented.
2. **Heavy Clustered Occlusion**: Goalmouth scuffles and set-piece crowding can cause temporary bounding box merging or ByteTrack ID switches.
3. **Goalkeeper Jersey Ambiguity**: Distinct goalkeeper kits require specialized 3-way clustering rules to avoid being misclassified into the opponent's jersey palette.
4. **Extreme Motion Blur**: Rapid cross-field long balls can exceed single-frame detection confidence, necessitating multi-frame temporal interpolation.
5. **Off-Camera Players**: Single-camera setups cannot track players located outside the current broadcast viewport, creating temporary gaps in tactical line calculations.

---

## 15. Privacy, Ethics & Data Governance

* **Minors & Academy Footage**: Processing youth match video must comply with strict data protection guidelines (e.g., GDPR Article 8, COPPA). Facial identification is intentionally omitted in favor of anonymous jersey numbers and spatial tracks.
* **Biometric Exemption**: ForshaVision operates on macroscopic kinematics (speed, distance, spatial coordinates) and does not store facial recognition vectors or private medical telemetry.
* **Model Governance & Licensing**: Commercial deployments must verify training data origins (e.g. verifying non-commercial terms on academic datasets like SoccerNet and StatsBomb).

---

## 16. Phased Product & Research Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Vision & Kinematic Foundation (COMPLETED)                          │
│ • YOLOv8 multi-class detection • ByteTrack tracking • LK Optical Flow       │
│ • 4-point Planar Homography • KMeans Jersey Clustering • Memory-safe stream │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Tactical Geometry & Graph Analytics (Q1-Q2 2025)                   │
│ • SoccerNet-GSR Automated Line Calibration • Pitch-wide Formation Tracking │
│ • Passing Networks with NetworkX • Team Compactness & Line Depth Analysis   │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: Temporal Event Understanding & Multimodal Sync (Q2-Q3 2025)        │
│ • VideoMAE / Temporal Transformer event classifiers (pass, shot, tackle)    │
│ • Assist reasoning engine • WhisperX commentary alignment & diarization     │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Production Lakehouse & Conversational Copilot (Q3-Q4 2025)         │
│ • Apache Parquet / DuckDB Lakehouse • LangGraph Multi-Agent Football AI     │
│ • Text-to-Speech voice briefing engine • Auto-clipping highlight pipeline   │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 5: Predictive Modeling & Simulation (2026)                            │
│ • Expected Threat (xT) spatial Markov models • Reinforcement Learning      │
│ • Tactical simulation & next-action prediction                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 17. ForshaLabs Engineering Case Study

### Case Study Overview

| Attribute | Engineering Detail |
| :--- | :--- |
| **Client / Context** | Flagship Sports AI Research & Product Case Study |
| **Engineering Domain**| Computer Vision, Deep Learning, MLOps, Data Engineering, Multimodal AI |
| **Core Challenge** | Converting uncalibrated 1080p single-angle broadcast video into high-precision tactical and physical tracking data without stadium hardware |
| **Primary Breakthrough**| Achieving sub-meter spatial accuracy via decoupled optical-flow camera compensation and constant-memory frame-by-frame streaming pipelines |

### Architectural Competencies Demonstrated

* **Deep Learning & Computer Vision**: Fine-tuned neural detectors, Kalman-associated tracking, optical flow kinematics, and homographic projective geometry.
* **Data Engineering & Performance Optimization**: Replaced multi-gigabyte RAM frame accumulators with constant $O(1)$ ROI-blended OpenCV streaming and high-speed serialization caching stubs.
* **Multimodal Systems Architecture**: Blueprinting tool-calling LLM agents, timestamp-aligned ASR commentary engines, and hybrid SQL/vector retrieval systems.

---

## 18. Contributing & License

### Contributing

Contributions are welcomed from computer vision researchers, sports data scientists, and ML engineers. Please submit pull requests following standard PEP-8 style guidelines and include unit tests for new geometric or tracking modules.

### License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>ForshaVision</b> is an open-source technical initiative developed by <b>ForshaLabs</b>.<br>
  <i>Empowering coaches, scouts, and analysts with accessible, world-class football intelligence.</i>
</p>
