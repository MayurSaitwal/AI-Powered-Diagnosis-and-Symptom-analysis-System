import pandas as pd
import numpy as np
import os
import re
from difflib import get_close_matches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
DISEASE_SYMPTOMS_PATH = os.path.join(DATA_PATH, "DiseaseAndSymptoms.csv")
DISEASE_PRECAUTION_PATH = os.path.join(DATA_PATH, "DiseasePrecaution.csv")


def _normalize_text(value):
    text = str(value).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_disease_name(value):
    return str(value).strip()

# Load datasets
disease_df = pd.read_csv(DISEASE_SYMPTOMS_PATH)

# Clean dataset
disease_df = disease_df.fillna("")

# Get all unique symptoms
all_symptoms = set()

for col in disease_df.columns[1:]:
    all_symptoms.update(disease_df[col].apply(_normalize_text))

all_symptoms.discard("")
all_symptoms = sorted(list(all_symptoms))

# Create symptom index
symptom_index = {symptom: i for i, symptom in enumerate(all_symptoms)}

# Build disease-level symptom statistics from all rows
disease_row_counts = {}
disease_symptom_counts = {}

for _, row in disease_df.iterrows():
    disease_name = _clean_disease_name(row.iloc[0])
    if not disease_name:
        continue

    if disease_name not in disease_symptom_counts:
        disease_symptom_counts[disease_name] = np.zeros(len(all_symptoms), dtype=float)
        disease_row_counts[disease_name] = 0

    disease_row_counts[disease_name] += 1

    for col in disease_df.columns[1:]:
        symptom = _normalize_text(row[col])
        if symptom in symptom_index:
            disease_symptom_counts[disease_name][symptom_index[symptom]] += 1

disease_names = list(disease_symptom_counts.keys())

# P(symptom | disease): how often a symptom appears for a disease across dataset rows
disease_prob_matrix = np.array([
    disease_symptom_counts[name] / max(disease_row_counts[name], 1)
    for name in disease_names
])

# Binary matrix: whether disease ever contains the symptom
disease_presence_matrix = (disease_prob_matrix > 0).astype(float)
disease_symptom_totals = disease_presence_matrix.sum(axis=1)

# Symptom rarity weight across diseases (IDF-like)
symptom_disease_counts = disease_presence_matrix.sum(axis=0)
symptom_weights = np.log((1 + len(disease_names)) / (1 + symptom_disease_counts)) + 1.0

precaution_lookup = {}
_precaution_file_mtime = None


def _refresh_precaution_lookup(force=False):
    """Reload precaution lookup if CSV changed so app reflects latest text updates."""
    global precaution_lookup, _precaution_file_mtime

    try:
        current_mtime = os.path.getmtime(DISEASE_PRECAUTION_PATH)
    except OSError:
        return

    if not force and _precaution_file_mtime == current_mtime:
        return

    precaution_df = pd.read_csv(DISEASE_PRECAUTION_PATH).fillna("")
    refreshed_lookup = {}
    for _, row in precaution_df.iterrows():
        disease = _clean_disease_name(row.iloc[0])
        if not disease:
            continue
        refreshed_lookup[_normalize_text(disease)] = row.iloc[1:].tolist()

    precaution_lookup = refreshed_lookup
    _precaution_file_mtime = current_mtime


_refresh_precaution_lookup(force=True)


# ---------------- PREDICTION ---------------- #

def predict_disease(input_symptoms):

    if not input_symptoms:
        return {"error": "No symptoms provided"}

    input_vector = np.zeros(len(all_symptoms), dtype=float)
    valid_symptoms = 0

    for symptom in input_symptoms:
        normalized = _normalize_text(symptom)
        matched_symptom = None

        if normalized in symptom_index:
            matched_symptom = normalized
        else:
            close_matches = get_close_matches(normalized, all_symptoms, n=1, cutoff=0.83)
            if close_matches:
                matched_symptom = close_matches[0]

        if matched_symptom:
            input_vector[symptom_index[matched_symptom]] = 1
            valid_symptoms += 1

    if valid_symptoms == 0:
        return {"error": "Symptoms not found in database"}

    input_mask = input_vector.astype(bool)
    input_weights = symptom_weights[input_mask]
    weighted_input_total = float(input_weights.sum())

    matched_presence = disease_presence_matrix[:, input_mask]
    matched_presence_count = matched_presence.sum(axis=1)

    matched_probabilities = disease_prob_matrix[:, input_mask]

    weighted_match = (matched_probabilities * input_weights).sum(axis=1)
    input_coverage = np.divide(
        weighted_match,
        max(weighted_input_total, 1e-9),
        out=np.zeros_like(weighted_match, dtype=float),
        where=True
    )

    non_input_mask = ~input_mask
    extra_symptom_mass = (
        disease_prob_matrix[:, non_input_mask] * symptom_weights[non_input_mask]
    ).sum(axis=1)

    soft_jaccard = np.divide(
        weighted_match,
        np.maximum(weighted_input_total + extra_symptom_mass, 1e-9),
        out=np.zeros_like(weighted_match, dtype=float),
        where=True
    )

    exact_overlap_ratio = np.divide(
        matched_presence_count,
        max(valid_symptoms, 1),
        out=np.zeros(len(disease_names), dtype=float),
        where=True
    )

    precision_like = np.divide(
        matched_presence_count,
        np.maximum(disease_symptom_totals, 1),
        out=np.zeros(len(disease_names), dtype=float),
        where=True
    )

    strong_match_ratio = np.divide(
        (matched_probabilities >= 0.5).sum(axis=1),
        max(valid_symptoms, 1),
        out=np.zeros(len(disease_names), dtype=float),
        where=True
    )

    similarities = (
        (0.42 * exact_overlap_ratio) +
        (0.20 * precision_like) +
        (0.20 * input_coverage) +
        (0.12 * soft_jaccard) +
        (0.06 * strong_match_ratio)
    )

    # Strongly reward diseases that cover every provided symptom when input has 3+ symptoms.
    full_cover_bonus = (exact_overlap_ratio >= 0.999) & (valid_symptoms >= 3)
    similarities = similarities + (0.20 * full_cover_bonus.astype(float))

    ranked_indices = similarities.argsort()[::-1]
    top_indices = [i for i in ranked_indices if similarities[i] > 0.04][:5]

    if not top_indices:
        top_indices = [i for i in ranked_indices if similarities[i] > 0][:3]

    if not top_indices:
        return {
            "error": "No meaningful disease match found for the given symptoms"
        }

    predictions = []
    for idx in top_indices:
        disease_name = disease_names[idx]
        raw_score = float(similarities[idx]) * 100
        score = round(max(0.0, min(raw_score, 100.0)), 2)
        predictions.append({
            "disease": disease_name,
            "confidence": score
        })

    top_disease = predictions[0]["disease"]
    confidence = predictions[0]["confidence"]

    precautions = get_disease_info(top_disease)

    severity = calculate_severity(input_symptoms)

    return {
        "error": None,
        "top_disease": top_disease,
        "confidence": confidence,
        "all_predictions": predictions,
        "precautions": precautions,
        "severity": severity
    }


def get_disease_info(disease):
    _refresh_precaution_lookup()
    key = _normalize_text(disease)
    precautions = precaution_lookup.get(key, [])
    precautions = [str(item).strip() for item in precautions if str(item).strip()]
    return precautions


def calculate_severity(symptoms):
    count = len(symptoms)

    if count <= 2:
        return "Low"
    elif count <= 5:
        return "Moderate"
    else:
        return "High"