# 🌾 SowSafe — Hyperlocal Monsoon Sowing Advisory

**Smart India Hackathon (SIH) 2026**  
**Problem Statement:** SIH26086 — *Hyperlocal Monsoon Onset & Break Prediction System (Block/Village Scale)*  
**Team:** Game Changer (ID 185376)  
**Category:** Agriculture, Food Tech & Rural Development / Software  

---

## 📌 Project Overview
A decision-first sowing advisory for rainfed farmers. A farmer picks **location + crop + irrigation** and gets back one clear answer: a **Sowing Safety Score (0–100)**, a **confidence level**, and a single **action** — *Sow Now · Wait · Sow with Irrigation · Change Crop* — with a plain-language reason, in **English / Hindi / Marathi**, with text-to-speech.

Rainfed agriculture makes up ~51% of India's net sown area, and for these farmers, **the single most important decision of the season is *when* to sow.** Sowing too early can result in a "false onset" — initial rains followed by a prolonged dry spell — killing germinating seeds and forcing costly re-sowing. 

SowSafe fuses onset probability, dry-spell risk beyond the crop's own tolerance, heavy-rain/waterlogging risk, and soil-moisture adequacy into one score, and turns it into an action a farmer can act on today.

## 🚀 Key Features
- **Sowing Safety Score (SSS):** A 0-100 metric calculated using crop-specific tolerance and irrigation availability.
- **Hyperlocal Precision:** Block / panchayat granularity instead of coarse regional forecasts.
- **False-Onset Guard:** Predicts dry spells that could destroy a newly sown crop.
- **Trilingual Support:** Available in English, Hindi, and Marathi with Text-to-Speech (TTS) integration.
- **Officer Dashboard:** Allows KVK scientists and agriculture officers to view block-level risk maps and endorse/override advisories.

## 📂 Project Structure
- [`sowsafe/`](sowsafe/) — The main source code for the prototype.
  - [`sowsafe/backend/`](sowsafe/backend/) — The FastAPI backend, ML prediction engine, and SSS logic.
  - [`sowsafe/frontend/`](sowsafe/frontend/) — The React Progressive Web App (PWA) for farmers and officers.
  - [`sowsafe/docs/`](sowsafe/docs/) — Architecture and technical documentation.
  - [`sowsafe/PRD.md`](sowsafe/PRD.md) — Comprehensive Product Requirements Document.
  - [`sowsafe/README.md`](sowsafe/README.md) — Detailed technical instructions for running the prototype locally.

## ⚙️ How to Run the Prototype
```bash
cd sowsafe/backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8077
```
Then open **http://127.0.0.1:8077**.

> **Note:** For full technical details, demo script, and architecture, please see the [technical README](sowsafe/README.md).

## 📊 System Architecture
SowSafe integrates data from IMD (gridded rainfall, AWS/ARG), satellite precipitation (GPM/IMERG), and global climate indices (ENSO/IOD/MJO) into a block-level XGBoost/baseline model. This is passed through an expert decision tree (using ICAR/KVK crop parameters) to output the SSS and final advisory.
