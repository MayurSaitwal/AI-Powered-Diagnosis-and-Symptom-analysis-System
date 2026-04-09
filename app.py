from flask import Flask, render_template, request, session, redirect, url_for, send_file
from utils.predictor import predict_disease, all_symptoms
import traceback
import os
import uuid
from datetime import datetime
from io import BytesIO
import pymysql
from pymysql import MySQLError
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------------------------------------------------
# Flask App Configuration
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

app.secret_key = os.getenv("SECRET_KEY", "india")
app.config["PROPAGATE_EXCEPTIONS"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": _safe_int(os.getenv("MYSQL_PORT", "3306"), 3306),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "ai_medical")
}


def _get_mysql_connection(include_database=True):
    config = DB_CONFIG.copy()
    config["cursorclass"] = pymysql.cursors.DictCursor
    if not include_database:
        config.pop("database", None)
    return pymysql.connect(**config)


def _initialize_mysql():
    """Create database/table if they do not exist so registration can persist."""
    try:
        bootstrap_conn = _get_mysql_connection(include_database=False)
        bootstrap_cursor = bootstrap_conn.cursor()
        bootstrap_cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        bootstrap_cursor.close()
        bootstrap_conn.close()

        conn = _get_mysql_connection(include_database=True)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_name VARCHAR(120) NOT NULL,
                email VARCHAR(190) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                age INT NOT NULL,
                gender VARCHAR(20) NOT NULL,
                city VARCHAR(100) NOT NULL,
                known_conditions VARCHAR(150) DEFAULT 'None',
                smoking VARCHAR(30) DEFAULT 'No',
                alcohol VARCHAR(30) DEFAULT 'No',
                emergency_contact VARCHAR(25) NOT NULL,
                consent_given TINYINT(1) NOT NULL DEFAULT 0,
                role VARCHAR(20) NOT NULL DEFAULT 'patient',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("[DB] MySQL initialized successfully")
    except MySQLError as db_error:
        print(f"[DB ERROR] Initialization failed: {db_error}")


def _friendly_db_error(db_error):
    if not db_error.args:
        return "Database error. Please try again."

    error_code = db_error.args[0]
    if error_code == 1045:
        return "Database authentication failed. Set correct MYSQL_USER and MYSQL_PASSWORD in AI/.env."
    if error_code == 1049:
        return "Database does not exist. Verify MYSQL_DATABASE or create it in MySQL Workbench."
    if error_code in (2003, 2005):
        return "Could not connect to MySQL server. Verify MYSQL_HOST and MYSQL_PORT and ensure MySQL is running."

    return "Database error. Please try again."


def _format_symptom_label(symptom):
    return str(symptom).replace("_", " ").strip().title()


def _get_patient_profile(email):
    if not email:
        return None

    try:
        conn = _get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT patient_name, email, age, gender, city,
                   known_conditions, smoking, alcohol, emergency_contact
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (email,)
        )
        profile = cursor.fetchone()
        cursor.close()
        conn.close()
        return profile
    except MySQLError as db_error:
        print(f"[DB ERROR] Profile lookup failed: {db_error}")
        return None


def _build_disease_insights(disease, confidence, severity, symptoms, precautions):
    symptom_count = len(symptoms or [])
    if confidence >= 75:
        certainty = "Strong pattern match"
    elif confidence >= 50:
        certainty = "Moderate pattern match"
    else:
        certainty = "Low-confidence signal"

    return [
        {
            "title": "Pattern Match",
            "value": certainty,
            "detail": f"Based on {symptom_count} matched symptom(s) for {disease}."
        },
        {
            "title": "Risk Signal",
            "value": severity,
            "detail": "Severity is estimated from symptom breadth and model confidence."
        },
        {
            "title": "Action Readiness",
            "value": "Precautions Ready",
            "detail": f"{len(precautions or [])} precaution item(s) generated for next steps."
        }
    ]


def _build_urgent_care_signals(symptoms, severity, confidence):
    symptoms = symptoms or []
    signal_map = {
        "chest_pain": "Chest pain should be reviewed urgently by a clinician.",
        "breathlessness": "Breathlessness can indicate respiratory distress and needs prompt assessment.",
        "slurred_speech": "Slurred speech can be neurological; seek immediate emergency support.",
        "weakness_of_one_body_side": "One-sided weakness can be a stroke warning sign.",
        "coma": "Coma is a medical emergency; contact emergency services immediately.",
        "high_fever": "Persistent high fever should be clinically evaluated, especially with fatigue or dehydration."
    }

    matched_signals = [
        message for key, message in signal_map.items() if key in symptoms
    ]

    if severity == "High" and confidence >= 65 and not matched_signals:
        matched_signals.append(
            "Your symptom combination appears high-risk. Same-day medical evaluation is recommended."
        )

    return matched_signals[:4]


def _group_precautions(precautions):
    categories = {
        "Immediate Care": [],
        "Home Care": [],
        "Lifestyle": [],
        "Follow-up": []
    }

    immediate_keywords = ["emergency", "urgent", "doctor", "hospital", "clinician", "consult"]
    home_keywords = ["rest", "hydrate", "water", "sleep", "diet", "fluid", "warm"]
    lifestyle_keywords = ["avoid", "smoking", "alcohol", "exercise", "stress", "hygiene"]

    for raw_item in precautions or []:
        item = str(raw_item).strip()
        if not item:
            continue
        text = item.lower()
        if any(word in text for word in immediate_keywords):
            categories["Immediate Care"].append(item)
        elif any(word in text for word in lifestyle_keywords):
            categories["Lifestyle"].append(item)
        elif any(word in text for word in home_keywords):
            categories["Home Care"].append(item)
        else:
            categories["Follow-up"].append(item)

    return {key: value for key, value in categories.items() if value}


def _build_urgency_payload(symptoms, severity, confidence):
    symptoms = symptoms or []
    severity_base = {
        "Low": 25,
        "Moderate": 55,
        "High": 78
    }

    critical_symptoms = {
        "chest_pain",
        "breathlessness",
        "slurred_speech",
        "weakness_of_one_body_side",
        "coma"
    }
    critical_hits = len([symptom for symptom in symptoms if symptom in critical_symptoms])

    score = severity_base.get(severity, 45)
    score += min(float(confidence or 0) * 0.22, 18)
    score += critical_hits * 8
    score = int(max(1, min(round(score), 100)))

    if score >= 85:
        band = "Critical"
        actions = [
            "Seek emergency medical care immediately.",
            "Do not self-medicate without clinical supervision.",
            "Keep an emergency contact informed right away."
        ]
    elif score >= 70:
        band = "High"
        actions = [
            "Arrange same-day doctor consultation.",
            "Monitor symptom changes every few hours.",
            "Avoid strenuous activity until reviewed."
        ]
    elif score >= 45:
        band = "Moderate"
        actions = [
            "Follow precautions and book a routine clinical review.",
            "Track symptoms for 24-48 hours.",
            "Escalate if symptoms worsen or new severe signs appear."
        ]
    else:
        band = "Low"
        actions = [
            "Continue supportive home care guidance.",
            "Maintain hydration, rest, and basic monitoring.",
            "Consult a clinician if symptoms persist."
        ]

    return {
        "score": score,
        "band": band,
        "actions": actions,
        "critical_hits": critical_hits
    }


def _build_confidence_breakdown(symptoms, predictions):
    normalized_input = []
    for item in symptoms or []:
        key = str(item).strip().lower().replace(" ", "_")
        if key and key not in normalized_input:
            normalized_input.append(key)

    matched = [symptom for symptom in normalized_input if symptom in all_symptoms]
    unmatched = [symptom for symptom in normalized_input if symptom not in all_symptoms]

    top_prediction = predictions[0] if predictions else {"confidence": 0}
    confidence_value = float(top_prediction.get("confidence", 0) or 0)
    ambiguity = "Low"
    if len(predictions) > 1:
        gap = confidence_value - float(predictions[1].get("confidence", 0) or 0)
        if gap < 8:
            ambiguity = "High"
        elif gap < 18:
            ambiguity = "Moderate"

    return {
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "matched_symptoms": matched[:8],
        "unmatched_symptoms": unmatched[:6],
        "ambiguity": ambiguity
    }


def _sanitize_symptom_input(symptoms_input):
    if not symptoms_input or not str(symptoms_input).strip():
        return [], "Please enter at least one symptom."

    parts = [chunk.strip().lower() for chunk in str(symptoms_input).split(",")]
    cleaned = []
    seen = set()
    for part in parts:
        if not part:
            continue
        normalized = part.replace(" ", "_")
        if len(normalized) < 2:
            continue
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)

    if not cleaned:
        return [], "Please provide valid symptom names separated by commas."
    if len(cleaned) > 25:
        return [], "Please provide up to 25 symptoms in one analysis for better quality."
    if len(cleaned) == 1 and cleaned[0] in {"pain", "fever", "cough"}:
        return [], "Please provide more specific symptoms for a meaningful analysis."

    return cleaned, None


def _build_history_metrics(history_items):
    symptom_counter = {}
    dates = []
    last_severity = "No data"

    for item in history_items or []:
        if item.get("severity"):
            last_severity = item.get("severity")

        for symptom in item.get("symptoms", []):
            symptom_counter[symptom] = symptom_counter.get(symptom, 0) + 1

        generated_at = str(item.get("generated_at", ""))
        try:
            dates.append(datetime.strptime(generated_at, "%Y-%m-%d %H:%M:%S").date())
        except ValueError:
            continue

    top_symptoms = sorted(
        symptom_counter.items(),
        key=lambda pair: pair[1],
        reverse=True
    )[:3]
    top_symptoms = [
        {"name": _format_symptom_label(name), "count": count}
        for name, count in top_symptoms
    ]

    check_streak = 0
    if dates:
        unique_dates = sorted(set(dates), reverse=True)
        check_streak = 1
        for idx in range(1, len(unique_dates)):
            delta = (unique_dates[idx - 1] - unique_dates[idx]).days
            if delta == 1:
                check_streak += 1
            else:
                break

    return {
        "top_symptoms": top_symptoms,
        "last_severity": last_severity,
        "check_streak": check_streak
    }


def _build_result_context_from_report(report_data):
    symptoms = report_data.get("symptoms", [])
    disease = report_data.get("disease", "Unknown")
    confidence = float(report_data.get("confidence", 0) or 0)
    severity = report_data.get("severity", "Moderate")
    predictions = report_data.get("predictions", [])
    precautions = report_data.get("precautions", [])

    return {
        "disease": disease,
        "confidence": confidence,
        "predictions": predictions,
        "precautions": precautions,
        "severity": severity,
        "symptoms": symptoms,
        "disease_insights": _build_disease_insights(disease, confidence, severity, symptoms, precautions),
        "urgent_care_signals": _build_urgent_care_signals(symptoms, severity, confidence),
        "urgency_payload": _build_urgency_payload(symptoms, severity, confidence),
        "confidence_breakdown": _build_confidence_breakdown(symptoms, predictions),
        "precaution_groups": _group_precautions(precautions)
    }


def _append_prediction_history(report_data):
    history_id = report_data.get("id") or uuid.uuid4().hex[:12]
    report_data["id"] = history_id

    history = session.get("prediction_history", [])
    history.append({
        "id": history_id,
        "disease": report_data.get("disease"),
        "confidence": report_data.get("confidence"),
        "severity": report_data.get("severity"),
        "generated_at": report_data.get("generated_at"),
        "symptom_count": len(report_data.get("symptoms", [])),
        "symptoms": report_data.get("symptoms", []),
        "predictions": report_data.get("predictions", []),
        "precautions": report_data.get("precautions", [])
    })
    session["prediction_history"] = history[-8:]
    session.modified = True


def _normalize_history_items(history_items):
    normalized_items = []
    for index, item in enumerate(history_items or []):
        normalized = dict(item)
        if not normalized.get("id"):
            normalized["id"] = f"legacy-{index}-{uuid.uuid4().hex[:6]}"
        normalized.setdefault("symptoms", [])
        normalized.setdefault("predictions", [])
        normalized.setdefault("precautions", [])
        normalized.setdefault("confidence", 0)
        normalized.setdefault("severity", "Moderate")
        normalized_items.append(normalized)
    return normalized_items


_initialize_mysql()


def _build_report_pdf(report_data):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4

    y = page_height - 50

    def write_line(text, size=11, bold=False, spacing=18):
        nonlocal y
        if y < 70:
            pdf.showPage()
            y = page_height - 50

        font_name = "Helvetica-Bold" if bold else "Helvetica"
        pdf.setFont(font_name, size)
        pdf.drawString(45, y, str(text))
        y -= spacing

    def write_wrapped(prefix, content, size=11):
        nonlocal y
        full_text = f"{prefix}{content}"
        words = full_text.split(" ")
        line = ""
        max_chars = 95

        for word in words:
            candidate = f"{line} {word}".strip()
            if len(candidate) <= max_chars:
                line = candidate
            else:
                write_line(line, size=size)
                line = word
        if line:
            write_line(line, size=size)

    write_line("AI MEDICAL TRIAGE - PREDICTION REPORT", size=15, bold=True, spacing=24)
    write_line(f"Generated At: {report_data.get('generated_at', '-')}", size=10, spacing=20)

    write_line("Summary", size=13, bold=True, spacing=20)
    write_wrapped("Top Prediction: ", report_data.get("disease", "-"))
    write_wrapped("Confidence: ", f"{report_data.get('confidence', '-')}%")
    write_wrapped("Severity: ", report_data.get("severity", "-"))

    write_line("", spacing=8)
    write_line("Symptoms Entered", size=13, bold=True, spacing=20)
    symptoms = report_data.get("symptoms", [])
    write_wrapped("Symptoms: ", ", ".join(symptoms) if symptoms else "-")

    write_line("", spacing=8)
    write_line("All Possible Predictions", size=13, bold=True, spacing=20)
    predictions = report_data.get("predictions", [])
    if predictions:
        for item in predictions:
            disease = item.get("disease", "-")
            conf = item.get("confidence", "-")
            write_wrapped("- ", f"{disease} ({conf}%)")
    else:
        write_line("- No predictions available")

    write_line("", spacing=8)
    write_line("Recommended Precautions", size=13, bold=True, spacing=20)
    precautions = report_data.get("precautions", [])
    if precautions:
        for item in precautions:
            write_wrapped("- ", item)
    else:
        write_line("- No precautions available")

    write_line("", spacing=10)
    write_wrapped(
        "Disclaimer: ",
        "This report is AI-generated and for informational purposes only. Consult a qualified doctor for diagnosis and treatment."
    )

    pdf.save()
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------
# Global Error Handler (Prevents 500 Crash)
# ---------------------------------------------------------

@app.errorhandler(404)
def handle_not_found(e):
    return render_template(
        "result.html",
        error="The requested page was not found."
    ), 404


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return render_template(
        "result.html",
        error="Method not allowed for this route."
    ), 405


@app.errorhandler(413)
def handle_payload_too_large(e):
    return render_template(
        "result.html",
        error="Request payload is too large. Please upload smaller data."
    ), 413


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return render_template(
        "result.html",
        error=e.description or "Request could not be completed."
    ), e.code


@app.errorhandler(Exception)
def handle_error(e):
    print("\n" + "=" * 60)
    print("APPLICATION ERROR:")
    print(str(e))
    print(traceback.format_exc())
    print("=" * 60 + "\n")

    return render_template("result.html",
                           error="Something went wrong. Please try again."), 500


# ---------------------------------------------------------
# Public Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/home")
def home():
    return render_template("dashboard.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/how_it_works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("about.html")


@app.route("/doctor_login")
def doctor_login():
    return render_template("login.html")


@app.route("/patient")
def patient():
    return render_template("patient.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/symptom_form")
def symptom_form():
    return render_template("register.html")


@app.route("/register_complete", methods=["POST"])
def register_complete():
    registration_data = {
        "patient_name": request.form.get("patient_name", "").strip(),
        "email": request.form.get("email", "").strip().lower(),
        "password": request.form.get("password", ""),
        "age": request.form.get("age", "0").strip(),
        "gender": request.form.get("gender", "").strip(),
        "city": request.form.get("city", "").strip(),
        "known_conditions": request.form.get("known_conditions", "None").strip(),
        "smoking": request.form.get("smoking", "No").strip(),
        "alcohol": request.form.get("alcohol", "No").strip(),
        "emergency_contact": request.form.get("emergency_contact", "").strip(),
        "role": request.form.get("role", "patient").strip(),
        "consent_given": 1 if request.form.get("consent") else 0
    }

    if not all([
        registration_data["patient_name"],
        registration_data["email"],
        registration_data["password"],
        registration_data["gender"],
        registration_data["city"],
        registration_data["emergency_contact"]
    ]):
        return render_template("register.html", error="Please fill all required fields."), 400

    try:
        age_value = int(registration_data["age"])
        if age_value < 1 or age_value > 120:
            return render_template("register.html", error="Age must be between 1 and 120."), 400
    except ValueError:
        return render_template("register.html", error="Invalid age value."), 400

    try:
        _initialize_mysql()

        conn = _get_mysql_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE email = %s", (registration_data["email"],))
        existing_user = cursor.fetchone()
        if existing_user:
            cursor.close()
            conn.close()
            return render_template("register.html", error="Email is already registered."), 409

        cursor.execute(
            """
            INSERT INTO users (
                patient_name, email, password_hash, age, gender, city,
                known_conditions, smoking, alcohol, emergency_contact,
                consent_given, role
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                registration_data["patient_name"],
                registration_data["email"],
                generate_password_hash(registration_data["password"]),
                age_value,
                registration_data["gender"],
                registration_data["city"],
                registration_data["known_conditions"],
                registration_data["smoking"],
                registration_data["alcohol"],
                registration_data["emergency_contact"],
                registration_data["consent_given"],
                registration_data["role"]
            )
        )
        conn.commit()
        cursor.close()
        conn.close()

        session["logged_in"] = True
        session["user_email"] = registration_data["email"]
        return render_template(
            "register_result.html",
            view=registration_data["patient_name"]
        )
    except MySQLError as db_error:
        print(f"[DB ERROR] Registration failed: {db_error}")
        return render_template("register.html", error=_friendly_db_error(db_error)), 500


@app.route("/download_report")
def download_report():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    report_data = session.get("last_report")
    if not report_data:
        return render_template("result.html", error="No report found. Please generate a prediction first.")

    pdf_buffer = _build_report_pdf(report_data)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="medical_prediction_report.pdf"
    )


# ---------------------------------------------------------
# Authentication Logic
# ---------------------------------------------------------

@app.route("/login_check", methods=["POST"])
def login_check():
    identifier = request.form.get("identifier", request.form.get("email", "")).strip()
    email = identifier.lower()
    password = request.form.get("password", "")

    if not identifier or not password:
        return render_template("login.html", error="Please enter username/email and password."), 400

    # Keep demo credentials as fallback.
    if email == "manas@gmail.com" and password == "12345":
        session["logged_in"] = True
        session["user_email"] = email
        return redirect(url_for("patient_dashboard"))

    try:
        conn = _get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT email, patient_name, password_hash
            FROM users
            WHERE email = %s OR LOWER(patient_name) = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (email, identifier.lower())
        )
        user = cursor.fetchone()
        if not user:
            cursor.close()
            conn.close()
            return render_template("login.html", error="Invalid username/email or password."), 401

        stored_password = str(user.get("password_hash", ""))

        # Backward compatibility: if a legacy plain-text password exists, allow login once and rehash.
        password_ok = False
        if stored_password.startswith("pbkdf2:") or stored_password.startswith("scrypt:"):
            password_ok = check_password_hash(stored_password, password)
        else:
            password_ok = stored_password == password

        if password_ok:
            if not (stored_password.startswith("pbkdf2:") or stored_password.startswith("scrypt:")):
                cursor.execute(
                    "UPDATE users SET password_hash = %s WHERE email = %s",
                    (generate_password_hash(password), email)
                )
                conn.commit()

            cursor.close()
            conn.close()
            session["logged_in"] = True
            session["user_email"] = user["email"]
            return redirect(url_for("patient_dashboard"))

        cursor.close()
        conn.close()
        return render_template("login.html", error="Invalid username/email or password."), 401
    except MySQLError as db_error:
        print(f"[DB ERROR] Login failed: {db_error}")
        return render_template("login.html", error=_friendly_db_error(db_error)), 500


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history/<history_id>")
def history_report(history_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    history = _normalize_history_items(session.get("prediction_history", []))
    selected = next((item for item in history if item.get("id") == history_id), None)

    if not selected:
        return render_template(
            "result.html",
            error="Selected analysis history was not found."
        )

    context = _build_result_context_from_report(selected)
    return render_template("result.html", **context, error=None)


@app.route("/history/<history_id>/delete", methods=["POST"])
def delete_history_item(history_id):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    history = _normalize_history_items(session.get("prediction_history", []))
    filtered_history = [item for item in history if item.get("id") != history_id]
    session["prediction_history"] = filtered_history

    last_report = session.get("last_report")
    if last_report and last_report.get("id") == history_id:
        session["last_report"] = filtered_history[-1] if filtered_history else None

    session.modified = True
    return redirect(url_for("patient_dashboard"))


# ---------------------------------------------------------
# Patient Dashboard
# ---------------------------------------------------------

@app.route("/patient_dashboard")
def patient_dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    user_email = session.get("user_email", "Patient")
    patient_profile = _get_patient_profile(user_email)
    normalized_history = _normalize_history_items(session.get("prediction_history", []))
    if normalized_history != session.get("prediction_history", []):
        session["prediction_history"] = normalized_history
        session.modified = True

    prediction_history = list(reversed(normalized_history))
    history_metrics = _build_history_metrics(normalized_history)
    recently_used_symptoms = []
    recently_used_symptom_entries = []
    if session.get("last_report"):
        raw_recent = session.get("last_report", {}).get("symptoms", [])[:8]
        recently_used_symptoms = [_format_symptom_label(item) for item in raw_recent]
        recently_used_symptom_entries = [
            {
                "label": _format_symptom_label(item),
                "value": str(item).strip().lower().replace(" ", "_")
            }
            for item in raw_recent
            if str(item).strip()
        ]

    if patient_profile:
        profile_payload = {
            "patient_name": patient_profile.get("patient_name") or user_email,
            "email": patient_profile.get("email") or user_email,
            "age": patient_profile.get("age") or "-",
            "gender": patient_profile.get("gender") or "-",
            "city": patient_profile.get("city") or "-",
            "known_conditions": patient_profile.get("known_conditions") or "None",
            "smoking": patient_profile.get("smoking") or "No",
            "alcohol": patient_profile.get("alcohol") or "No"
        }
    else:
        profile_payload = {
            "patient_name": user_email,
            "email": user_email,
            "age": "-",
            "gender": "-",
            "city": "-",
            "known_conditions": "None",
            "smoking": "No",
            "alcohol": "No"
        }

    return render_template(
        "patient_dashboard.html",
        username=user_email,
        patient_profile=profile_payload,
        prediction_history=prediction_history,
        recent_history_count=len(prediction_history),
        recently_used_symptoms=recently_used_symptoms,
        recently_used_symptom_entries=recently_used_symptom_entries,
        history_metrics=history_metrics,
        available_symptoms=all_symptoms,
        symptom_count=len(all_symptoms)
    )


# ---------------------------------------------------------
# Prediction Route
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        if not session.get("logged_in"):
            return redirect(url_for("login"))

        symptoms_input = request.form.get("symptoms", "")
        symptoms_list, validation_error = _sanitize_symptom_input(symptoms_input)

        if validation_error:
            return render_template("result.html",
                                   error=validation_error)

        print(f"\n[APP] Symptoms Received: {symptoms_list}")

        result = predict_disease(symptoms_list)

        if result["error"]:
            return render_template("result.html",
                                   error=result["error"])

        print(f"[APP] Top Prediction: {result['top_disease']} "
              f"({result['confidence']}%)")

        session["last_report"] = {
            "disease": result["top_disease"],
            "confidence": result["confidence"],
            "predictions": result["all_predictions"],
            "precautions": result["precautions"],
            "severity": result["severity"],
            "symptoms": symptoms_list,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        _append_prediction_history(session["last_report"])

        context = _build_result_context_from_report(session["last_report"])
        return render_template("result.html", **context, error=None)

    except Exception as e:
        print("[PREDICT ERROR]")
        print(str(e))
        print(traceback.format_exc())

        return render_template("result.html",
                               error="Prediction failed. Please try again.")


# ---------------------------------------------------------
# Run Server
# ---------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = _safe_int(os.getenv("PORT", "5000"), 5000)
    app.run(
        debug=False,
        host=host,
        port=port,
        threaded=True
    )