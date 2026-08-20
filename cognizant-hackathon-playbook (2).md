# Cognizant Hackathon — Execution Playbook

*Compiled from the kickoff briefing: the evaluation criteria, the "expectation of best solution" slide, and the 2023 reference project they showed. Originally written to apply to any of the 16 use cases — now that the allocation landed, §0 below has our specific assignment.*

---

## 0. Confirmed Assignment

**Team (Batch 1):** Adharsh, Aditya B Pillai, Alfred Devy, Devarth Raj, John Paul, Lakshmidath S, Muzammil CK, Pranav P P
**College mentor:** Aby Abahai T · **Cognizant mentor:** Jentil Jose

**Use case: Student Placement Prediction System** (Classification)
*Data source: Kaggle – Campus Recruitment/Placement dataset*

Placement cells track students across spreadsheets and only learn who's unprepared once companies start shortlisting — marks, backlogs, certifications, project work, mock-test scores, and attendance sit in separate systems with no single readiness view per student or department.

**MVP:** engineer features from marks, backlogs, skills, and attendance; compare Logistic Regression, Random Forest, and XGBoost; target ROC-AUC of 0.80+. Deploy a Streamlit or Power BI dashboard showing placement likelihood, per-student skill gaps, and department trends.
**Stretch:** cohort what-if simulation.

This is Classification — use the Classification block in §5 directly (LR/RF/XGBoost comparison for B, feature-engineering pipeline for E, ROC-AUC + confusion matrix for G).

Worth being honest about: this is one of the most commonly attempted student hackathon projects — the idea itself won't differentiate. The alternatives table (B), per-student/department dashboard depth, and the what-if stretch goal are where this can actually stand out against other teams' placement predictors.

---

## 1. What Cognizant Is Actually Grading

The rubric isn't "did you build something that works." It's structured as a full delivery lifecycle, and every stage is scored on its own. Read together, the three briefing slides say:

| Dimension | What they're checking | Where it shows up |
|---|---|---|
| Use Case Understanding | Do you actually understand the problem, not just skim it | Problem write-up, answers in mentor connects |
| Solution Architecture | Is the design sound and feasible, do components integrate cleanly | Architecture diagram, design doc |
| Innovation & Creativity | Original thinking, non-obvious AI/ML use — not "call an API and stop" | Idea-formation section, model justification |
| UI/UX | Usability, navigation, visual polish | The actual product screens |
| Code Quality | Readability, best practices, documentation | The repo |
| Model Performance | Accuracy/precision/recall/F1 or the task-appropriate metric | Results section, real numbers and curves |
| Deployment & Integration | CI/CD, cloud deployment, API integration — not "runs on my laptop" | Deployed link, deployment write-up |
| Presentation & Communication | Clarity, structure, defending choices live | Final presentation + Q&A |
| Collaboration & Teamwork | Task distribution, how the team actually worked | Roadmap/task breakdown, observed across the week |

Three lines in the post-build guideline change *how we work*, not just what we submit at the end:

- **"Proposed architecture and various alternatives"** / **"Integration alternatives of the solution"** — they explicitly want to see more than one approach considered, with a reason the others were dropped. One architecture with no comparison loses points even if it's the right call.
- **"Attendance in various mentor connects"** is a graded line, not a courtesy. It's scored as a process across the week, not a single Friday demo — skipping sessions costs points independent of the final product.
- **"Breadth of sample data"** and **"Real time decisions"** — the demo shouldn't run on one cherry-picked input. Show it handling a spread of cases, and live where possible, not only pre-baked output.

Final deliverables, stated directly on their "Expectation of Best Solution" slide:
1. Architecture + source code
2. User Interface
3. Documentation + video
4. Estimation of development + roadmap
5. Presentation

Treat these as five separate, non-negotiable artifacts — not one bundled "project." Missing any one at submission is a gap against the rubric regardless of how good the model is.

---

## 2. The Pattern Behind Their Own Reference Example

They used their 2023 project (TumorVision, brain tumor detection) as the calibration example. It maps almost one-to-one onto the rubric above, and it's the clearest signal of how much depth "good" actually means at each step:

| What TumorVision did | Rubric line it satisfies |
|---|---|
| Wrote out the problem, concrete tasks, known challenges, and target metrics before building anything | Use Case Understanding |
| Compared object detection vs. image segmentation in a table against their own stated requirements, then justified segmentation | Innovation, Architecture (alternatives) |
| Explicitly listed dependencies: data source (MRI scans), compute (GPU via Colab), storage (S3 + local), frameworks (TensorFlow/PyTorch) | Architecture Consideration |
| Split the stack into clean layers: frontend (HTML/CSS/JS/Bootstrap), backend (Django), database (SQLite + S3), deployment (SageMaker) | Solution Architecture, Deployment |
| One clear diagram: user → UI → S3 → two branches (U-Net segmentation, fine-tuned ViT classification) → generated report | Architecture, Source Code |
| Built a branded, multi-screen product — login, landing page, dashboard, detection form, results view — not a bare notebook | UI/UX |
| Reported real numbers: test image / ground truth / prediction side by side, training vs. validation accuracy and loss curves, final accuracy (96%) | Model Performance and Evaluation |

The takeaway: every rubric line became one concrete artifact. That's the structure to copy, whichever use case we pick.

---

## 3. Generic Build Template

Fill each of these in for whichever use case we choose. This is the backbone of both the final doc and the presentation.

**A. Problem statement detailing** — Before any code: the core problem in 2–3 sentences, the concrete tasks, known challenges specific to the chosen dataset, and the exact metrics we'll report at the end.

**B. Idea formation — alternatives table** — For the core technical decision, a table: our requirement → candidate approach → why it wins over the alternative. Minimum two approaches compared. This is the cheapest, highest-leverage artifact for "Innovation and Creativity" — it directly answers a named rubric line.

**C. Solution & dependencies** — Bullet list: data source, compute needed, storage/infrastructure, frameworks/libraries.

**D. Tech stack** — One table or diagram: frontend / backend / database / deployment, with the specific technology under each.

**E. Architecture diagram** — One page, legible at a glance: user → UI → storage → processing/model → output → back to user.

**F. UI** — Minimum: a working input flow, a results view, and a home/dashboard screen so it reads as a product, not a script. Name it — a named, branded tool reads as more finished than an unlabeled form.

**G. Results** — The metric the use case's own brief specifies (each of the 16 already states a target number), a visual (curve, confusion matrix, before/after, sample predictions), and one honest sentence on where it falls short.

---

## 4. Day-by-Day Plan (7 days, 8 people)

*Default shape — adjust once roles are actually split.*

- **Day 1 — Receive + align.** Use case is randomly allocated, not chosen — no selection phase to spend time on. The moment it lands, assign workstreams (see §5) matched to whatever type you got. Write sections A and B together — these need whole-team input regardless of whether the problem was picked or assigned. First mentor connect: bring the alternatives table, and ask whether scoring is calibrated per use case or judged head-to-head — worth knowing early.
- **Day 2 — Design + data.** Lock architecture diagram and tech stack (C, D, E). In parallel, whoever owns data starts EDA/cleaning on the real dataset. Repo scaffolded, branching agreed.
- **Day 3–4 — Core build.** Model/backend and frontend scaffolding run in parallel, not sequentially — frontend builds against a mocked response shape while the model trains. This is where 8 people matters: don't let six wait on two.
- **Day 5 — Integration.** Wire real model output into the real UI. Start deployment (CI/CD + cloud) here, not on day 7 — deployment problems always take longer than expected.
- **Day 6 — Evaluate + document.** Run against a spread of sample inputs, not one demo case. Capture metrics and visuals for section G. Draft documentation, record the video, write the roadmap/estimate.
- **Day 7 — Rehearse + present.** Full run-through with Q&A practice. "Ability to answer questions and justify design choices" is graded live — the team should be able to defend the alternatives *not* chosen in section B under questioning.

---

## 5. Quick-Adapt Notes by Use-Case Type

The 16 options cluster into a handful of shapes. Once the allocation lands, use the matching block for what B, E, and G specifically contain.

**Classification** (placement, dropout, disease risk, churn, crop advisor, fake news detection)
- B: compare Logistic Regression / Random Forest / XGBoost against each other — for text ones (fake news) compare TF-IDF + classical vs. fine-tuned BERT
- E: data → feature engineering → model → dashboard (text ones: raw text → vectorization → model)
- G: the specific target metric the brief states, plus a confusion matrix

**Explainable AI** (loan approval)
- B must include why SHAP/LIME over a black-box-only model
- E: add an explicit "explanation layer" between model and output
- G: per-decision reason codes shown in the UI, not just an aggregate accuracy number

**Time series / forecasting** (retail sales, traffic congestion, energy consumption)
- B: statistical model (ARIMA/SARIMA/Prophet) vs. ML model (LightGBM) vs. a naive baseline — the naive-baseline comparison is what the brief is actually asking for
- G: forecast vs. actual plot over a hold-out period, MAPE/RMSE against the baseline

**Regression / Optimization** (waste management)
- B: prediction approach vs. how it feeds the downstream optimization/routing decision
- G: MAE against actuals, plus the resulting optimization output (a route or priority list), not just the raw prediction

**NLP / Information extraction** (resume screening)
- B: keyword/TF-IDF matching vs. embedding-based semantic matching
- G: precision on a manually labeled sample, example outputs showing matched vs. missing skills

**GenAI / NLP** (interview prep companion, text-to-SQL)
- B: prompt-engineering-only vs. retrieval-augmented/fine-tuned approach
- E: guardrails shown explicitly (read-only execution for SQL, sanity-checking for interview scoring)
- G: execution accuracy or rated score against the brief's stated target

**GenAI / RAG** (enterprise knowledge assistant)
- B: chunking/embedding strategy comparison, and what happens on weak retrieval (refuse vs. hallucinate)
- E: the refusal path belongs in the architecture diagram, not just the happy path
- G: hallucination rate and answer accuracy on the gold Q&A set — named directly in the brief, easiest rubric hit on the list

**Anomaly detection** (cybersecurity threat detection)
- B: supervised classifier vs. Isolation Forest/autoencoder — justify against severe class imbalance
- G: recall specifically on rare attack classes (overall accuracy is misleading under imbalance), plus a near-real-time scoring demo

---

## 6. Final Deliverables Checklist

- [ ] Architecture diagram (one page, legible)
- [ ] Source code in a clean repo (readable, documented, best practices)
- [ ] Working UI (branded, multi-screen, not a bare script)
- [ ] Written documentation (problem, approach, alternatives considered, results)
- [ ] Demo video
- [ ] Development estimate / roadmap (built vs. "next")
- [ ] Presentation deck, rehearsed with anticipated Q&A
- [ ] Deployed, not just local (CI/CD + cloud, even a minimal setup)
- [ ] Results shown across a spread of sample inputs, not one cherry-picked case

## 7. Mentor Connects — Treat as Graded, Because They Are

- Bring something concrete to show or ask at every session — graded attendance rewards visible engagement, not silent presence.
- Bring the alternatives table (section B) to the first connect and get a reaction before sinking engineering time into it — cheaper to redirect on day 1 than day 4.
- Surface architecture and metric decisions early, so the final presentation isn't the first time anyone outside the team has seen the design.
