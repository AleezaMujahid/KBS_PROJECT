# 🏭 Industrial Fault Diagnosis – Knowledge-Based System

**Course:** CT-359 – Knowledge-Based Systems  
**Project Type:** Rule-Based Expert System  
**Interface:** Streamlit  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset

## 📌 Project Overview

Industrial Fault Diagnosis is a Knowledge-Based System (KBS) designed to identify potential faults in industrial machines using sensor readings.

The system uses a predefined knowledge base, IF–THEN rules, forward chaining, backward chaining, and certainty factors to diagnose faults and recommend appropriate maintenance actions.

Unlike Machine Learning systems, it relies on expert knowledge and logical reasoning rather than training a predictive model.

## 🎯 Objectives

- Identify potential industrial machine faults.
- Represent expert knowledge using facts and IF–THEN rules.
- Apply forward and backward chaining.
- Calculate confidence values using certainty factors.
- Generate maintenance recommendations.
- Provide explainable diagnostic results.

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

- 10,000 records.
- Machine sensor readings.
- Five failure modes.
- Includes air temperature, process temperature, rotational speed, torque, and tool wear.

**Dataset:** [UCI AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- JSON
- PyVis
- Pytest / Python unittest

## 🧠 How It Works

1. **Sensor Input:** Collects machine readings from the user.
2. **Preprocessing:** Converts sensor values into symptoms using predefined thresholds.
3. **Knowledge Base:** Stores machine faults, symptoms, and diagnostic rules.
4. **Inference Engine:** Uses forward and backward chaining to derive possible faults.
5. **Certainty Factors:** Calculates confidence values for diagnostic results.
6. **Recommendation Engine:** Suggests and ranks maintenance actions.
7. **Explanation:** Displays fired rules, supporting evidence, and reasoning graphs.

## 🏗️ System Architecture

The system follows a modular Knowledge-Based System architecture consisting of the following components:

1. **User Interface:** Streamlit provides an interactive interface for entering sensor readings and viewing diagnostic results.

2. **Preprocessing Module:** Processes raw sensor readings and converts them into meaningful symptoms using predefined thresholds.

3. **Knowledge Base:** Stores machine-related facts, symptoms, fault descriptions, and IF–THEN diagnostic rules.

4. **Inference Engine:** Applies forward chaining and backward chaining to identify possible machine faults based on available evidence.

5. **Certainty Factor Module:** Calculates and combines confidence values to determine the strength of possible diagnoses.

6. **Recommendation Engine:** Ranks maintenance recommendations according to stakeholder priorities and fault severity.

7. **Explanation Module:** Explains the reasoning behind diagnoses by displaying activated rules, supporting symptoms, and reasoning graphs.

## 🖥️ Key Features

- Interactive Streamlit interface.
- Manual sensor readings and sample machine scenarios.
- Automatic symptom detection.
- Follow-up questions for missing information.
- Ranked fault diagnosis with confidence values.
- Stakeholder-based maintenance recommendations.
- Rule-based explanations and reasoning graphs.
- What-if analysis using backward chaining.
- Support for multiple fault scenarios.

## 📁 Project Structure
```text
KBS_PROJECT/
├── data/
├── scripts/
├── src/
├── tests/
│
├── .gitignore
├── app.py
├── evaluation_output.txt
├── README.md
└── requirements.txt
```

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/AleezaMujahid/KBS_PROJECT.git
cd KBS_PROJECT
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🧪 Testing & Evaluation

Run individual component tests:

**Knowledge Base and Preprocessing**

```bash
python tests/test_person_a.py
```

**Inference Engine and Reasoning**

```bash
python tests/test_person_b.py
```

**Streamlit Interface**

```bash
python tests/test_person_c.py
```

**Run System Evaluation**

```bash
python scripts/evaluate.py
```

The evaluation measures diagnostic performance using Top-1 and Top-3 accuracy against labeled test cases.

## 📚 KBS Concepts

- Knowledge Representation
- Rule-Based Systems
- Forward Chaining
- Backward Chaining
- Certainty Factors
- Conflict Resolution
- Explainable Reasoning
- Decision Support Systems

## 📌 Conclusion

This project demonstrates how a rule-based Knowledge-Based System can use expert knowledge and logical reasoning to diagnose industrial machine faults and support maintenance decisions through an interactive and explainable interface.
