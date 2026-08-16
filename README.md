# Student Placement Prediction

A Machine Learning project that predicts whether a student is likely to be placed based on academic performance, technical skills, soft skills, internships, projects, work experience, certifications, attendance and backlogs.

## 📌 Project Overview

Student placement depends on several academic and professional factors. This project uses Machine Learning to analyze student-related features and predict their placement status.

The system follows a complete Machine Learning pipeline:

Raw Dataset → Data Analysis → Visualization → Preprocessing → Model Training → Evaluation → Prediction

## 🎯 Objective

The main objective of this project is to build a machine learning model that can predict:

- `0` → Not Placed
- `1` → Placed

The model can help identify students who may require additional training or placement preparation.

---

## 📊 Dataset

The dataset contains **5,000 student records** and initially contains 18 columns.

### Features

| Feature | Description |
|---|---|
| gender | Student gender |
| ssc_percentage | Secondary school percentage |
| hsc_percentage | Higher secondary percentage |
| degree_percentage | Degree percentage |
| cgpa | College CGPA |
| entrance_exam_score | Entrance examination score |
| technical_skill_score | Technical skill assessment score |
| soft_skill_score | Soft skill assessment score |
| internship_count | Number of internships |
| live_projects | Number of live projects |
| work_experience_months | Previous work experience |
| certifications | Number of certifications |
| attendance_percentage | Attendance percentage |
| backlogs | Number of academic backlogs |
| extracurricular_activities | Participation in extracurricular activities |
| placement_status | Placement outcome |

### Removed Columns

The following columns were removed during preprocessing:

- `student_id` — identifier only
- `salary_package_lpa` — removed to prevent data leakage because salary is known after placement

---

## 🔎 Exploratory Data Analysis

The dataset was analyzed for:

- Missing values
- Duplicate records
- Unique values
- Statistical distributions
- Placement distribution
- Feature averages by placement status
- Gender vs placement
- Extracurricular activities vs placement
- Feature correlations

### Data Quality

- Total records: **5,000**
- Missing values: **0**
- Duplicate rows: **0**

### Placement Distribution

- Not Placed: **4,134 (82.68%)**
- Placed: **866 (17.32%)**

The target variable is therefore imbalanced.

---

## 📈 Data Visualization

The project generates visualizations for important relationships between features and placement status.

Examples include:

- Placement distribution
- CGPA vs placement
- Technical skills vs placement
- Soft skills vs placement
- Backlogs vs placement
- Internships vs placement
- Live projects vs placement
- Work experience vs placement
- Attendance vs placement
- Gender vs placement
- Extracurricular activities vs placement
- Correlation heatmap

All generated visualizations are stored in:

```text
visualizations/