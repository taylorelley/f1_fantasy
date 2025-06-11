# app.py

import os
import json
import traceback
import pandas as pd
import numpy as np
import re
import io
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from authlib.integrations.flask_client import OAuth

from f1_optimizer import (
    F1VFMCalculator,
    F1TrackAffinityCalculator,
    F1TeamOptimizer,
    get_races_completed,
    get_expected_race_pace,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["DEFAULT_DATA_FOLDER"] = "default_data"
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)
oauth.register(
    name="facebook",
    client_id=os.environ.get("FACEBOOK_CLIENT_ID"),
    client_secret=os.environ.get("FACEBOOK_CLIENT_SECRET"),
    access_token_url="https://graph.facebook.com/v12.0/oauth/access_token",
    access_token_params=None,
    authorize_url="https://www.facebook.com/v12.0/dialog/oauth",
    authorize_params=None,
    api_base_url="https://graph.facebook.com/v12.0/",
    client_kwargs={"scope": "email"},
)
oauth.register(
    name="github",
    client_id=os.environ.get("GITHUB_CLIENT_ID"),
    client_secret=os.environ.get("GITHUB_CLIENT_SECRET"),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "user:email"},
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["DEFAULT_DATA_FOLDER"], exist_ok=True)

sessions = {}

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    provider_id = db.Column(db.String(256), unique=True, nullable=False)
    provider = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(256))
    email = db.Column(db.String(256))
    username = db.Column(db.String(256), unique=True)
    password_hash = db.Column(db.String(256))
    admin = db.Column(db.Boolean, default=False)
    config_json = db.Column(db.Text, default="{}")

class ScheduledOptimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    config_json = db.Column(db.Text, nullable=False)
    scheduled_for = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='pending')
    result_json = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



def get_data_folder(session_id=None):
    if session_id and session_id in sessions:
        folder = sessions[session_id]["folder"]
        return folder if folder.endswith("/") else folder + "/"
    elif has_default_data():
        folder = app.config["DEFAULT_DATA_FOLDER"]
        return folder if folder.endswith("/") else folder + "/"
    return None


def has_default_data():
    required_files = [
        "driver_race_data.csv",
        "constructor_race_data.csv",
        "calendar.csv",
        "tracks.csv",
    ]
    base = app.config["DEFAULT_DATA_FOLDER"]
    for fname in required_files:
        if not os.path.exists(os.path.join(base, fname)):
            return False
    return True


def has_driver_mapping():
    mapping_path = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
    return os.path.exists(mapping_path)


def load_settings():
    """Load optimisation parameter settings from settings.json"""
    settings_path = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "settings.json")
    defaults = {
        "outlier_stddev_factor": 2.0,
        "trend_slope_threshold": 1.7,
        "recent_races_fraction": 0.4,
        "long_term_weight": 0.7,
        "interaction_weight": 0.5,
        "top_n_candidates": 10,
        "use_ilp": False,
        "cron_schedule": "0 18 * * 6",
        "smtp_host": "",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": "",
        "smtp_tls": True,
        "smtp_from": "",
    }
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                if k == "use_ilp":
                    defaults[k] = bool(v)
                elif k in ("top_n_candidates", "smtp_port"):
                    defaults[k] = int(v)
                elif k in ("cron_schedule", "smtp_host", "smtp_username", "smtp_password", "smtp_from"):
                    defaults[k] = str(v)
                elif k == "smtp_tls":
                    defaults[k] = bool(v)
                else:
                    defaults[k] = float(v)
        except Exception:
            pass
    return defaults


def save_settings(data):
    """Save optimisation parameter settings to settings.json"""
    settings_path = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "settings.json")
    try:
        with open(settings_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def load_default_data():
    if not has_default_data():
        return None
    try:
        base = app.config["DEFAULT_DATA_FOLDER"] + "/"
        races_completed = get_races_completed(base)

        driver_df = pd.read_csv(os.path.join(base, "driver_race_data.csv"))
        constructor_df = pd.read_csv(os.path.join(base, "constructor_race_data.csv"))

        drivers_list = sorted(driver_df["Driver"].astype(str).unique().tolist())
        constructors_list = sorted(constructor_df["Constructor"].astype(str).unique().tolist())

        return {
            "races_completed": races_completed,
            "drivers": drivers_list,
            "constructors": constructors_list,
            "has_driver_mapping": has_driver_mapping(),
        }
    except Exception:
        return None


scheduler = BackgroundScheduler()

def send_email(to_email, subject, html_body, settings):
    if not settings.get("smtp_host"):
        print("SMTP host not configured; skipping email")
        return False
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = settings.get("smtp_from") or settings.get("smtp_username") or settings.get("smtp_host")
    msg["To"] = to_email
    try:
        server = smtplib.SMTP(settings.get("smtp_host"), settings.get("smtp_port"))
        if settings.get("smtp_tls", True):
            server.starttls()
        if settings.get("smtp_username"):
            server.login(settings.get("smtp_username"), settings.get("smtp_password"))
        server.sendmail(msg["From"], [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Failed to send email", e)
        return False


def perform_optimization(data, user=None):
    session_id = data.get("session_id", "default")
    if session_id == "default":
        if not has_default_data():
            raise ValueError("No default data; upload files first.")
        data_folder = get_data_folder("default")
        default_info = load_default_data()
        races_completed = default_info["races_completed"]
    else:
        if session_id not in sessions:
            raise ValueError("Session not found; upload files first.")
        data_folder = get_data_folder(session_id)
        races_completed = sessions[session_id]["races_completed"]

    use_fp2 = bool(data.get("use_fp2_pace", False))
    raw_pw = data.get("pace_weight", None)
    pace_weight = float(raw_pw) if raw_pw is not None else 0.25
    pace_modifier_type = data.get("pace_modifier_type") or "conservative"

    cal_path = os.path.join(data_folder, "calendar.csv")
    cal_df = pd.read_csv(cal_path, skipinitialspace=True)
    next_race = f"Race{races_completed + 1}"
    row = cal_df[cal_df["Race"] == next_race]
    if use_fp2:
        if row.empty or "meeting_key" not in cal_df.columns or pd.isna(row.iloc[0]["meeting_key"]):
            use_fp2 = False
            meeting_key = None
            race_year = None
        else:
            meeting_key = int(row.iloc[0]["meeting_key"])
            race_year = int(row.iloc[0]["year"])
    else:
        meeting_key = None
        race_year = None

    config = {
        "base_path":            data_folder,
        "races_completed":      races_completed,
        "current_drivers":      data.get("current_drivers", []),
        "current_constructors": data.get("current_constructors", []),
        "remaining_budget":     float(data.get("remaining_budget", 0.0)),
        "step1_swaps":          int(data.get("step1_swaps", 0)),
        "step2_swaps":          int(data.get("step2_swaps", 0)),
        "weighting_scheme":     data.get("weighting_scheme", "trend_based"),
        "risk_tolerance":       data.get("risk_tolerance", "medium"),
        "multiplier":           int(data.get("multiplier", 1)),
        "use_parallel":         False,
        "use_fp2_pace":         use_fp2,
        "pace_weight":          pace_weight,
        "pace_modifier_type":   pace_modifier_type,
        "next_meeting_key":     meeting_key,
        "next_race_year":       race_year,
    }

    settings = load_settings()
    config.update(settings)

    if user is not None:
        try:
            user.config_json = json.dumps(config)
            db.session.commit()
        except Exception:
            pass

    results = {"status": "running", "progress": []}

    if config["use_fp2_pace"]:
        results["progress"].append(f"Fetching FP2 pace data for meeting_key {meeting_key}...")
    results["progress"].append("Calculating VFM scores...")
    vfm_calc = F1VFMCalculator(config)
    driver_vfm_df, constructor_vfm_df = vfm_calc.run()
    results["progress"].append("VFM calculation complete")

    actual_fp2_applied = False
    if config["use_fp2_pace"] and "Pace_Score" in driver_vfm_df.columns:
        if driver_vfm_df["Pace_Score"].sum() > 0:
            actual_fp2_applied = True

    results["progress"].append("Calculating track affinities...")
    affinity_calc = F1TrackAffinityCalculator(config)
    driver_aff_df, constructor_aff_df = affinity_calc.run()
    results["progress"].append("Track affinity calculation complete")

    results["progress"].append("Optimizing team selection...")
    optimizer = F1TeamOptimizer(config)
    if not optimizer.load_data():
        raise ValueError("Error loading data for optimization")
    best_dict, base_s1, base_s2 = optimizer.run_dual_step_optimization()

    step1 = best_dict.get("step1_result")
    step2 = best_dict.get("step2_result")

    resp = {
        "status": "complete",
        "success": True,
        "optimization": {
            "step1": {
                "race":             optimizer.step1_race,
                "circuit":          optimizer.step1_circuit,
                "swaps":            step1["swaps"] if step1 else [],
                "expected_points":  step1["points"] if step1 else base_s1[0],
                "improvement":      (step1["points"] - base_s1[0]) if step1 else 0.0,
                "boost_driver":     step1["boost_driver"] if step1 else base_s1[3],
                "team": {
                    "drivers":      step1["drivers"] if step1 else config["current_drivers"],
                    "constructors": step1["constructors"] if step1 else config["current_constructors"],
                },
            },
            "step2": {
                "race":             optimizer.step2_race,
                "circuit":          optimizer.step2_circuit,
                "swaps":            step2["swaps"] if step2 else [],
                "expected_points":  step2["points"] if step2 else base_s2[0],
                "improvement":      (step2["points"] - base_s2[0]) if step2 else 0.0,
                "boost_driver":     step2["boost_driver"] if step2 else base_s2[3],
                "team": {
                    "drivers":      step2["drivers"] if step2 else config["current_drivers"],
                    "constructors": step2["constructors"] if step2 else config["current_constructors"],
                },
                "budget_used":      step2["cost"] if step2 else base_s2[2],
                "budget_remaining": round(optimizer.max_budget - (step2["cost"] if step2 else base_s2[2]), 2),
            },
            "summary": {
                "total_improvement":  round(best_dict["final_points"] - base_s2[0], 2),
                "patterns_evaluated": optimizer.performance_stats["patterns_evaluated"],
                "optimization_time":  round(optimizer.performance_stats["optimization_time"], 2),
                "step1_time":        round(optimizer.performance_stats["step1_time"], 2),
                "step2_time":        round(optimizer.performance_stats["step2_time"], 2),
            },
        },
        "progress": results["progress"],
    }

    if actual_fp2_applied:
        pace_info = {
            "meeting_key":       meeting_key,
            "year":              race_year,
            "pace_weight":       config["pace_weight"],
            "modifier_type":     config["pace_modifier_type"],
            "applied":           True,
            "pace_adjustments":  [],
        }
        for drv in config["current_drivers"]:
            row = optimizer.drivers_df[optimizer.drivers_df["Driver"] == drv]
            if not row.empty:
                ps = row.iloc[0].get("Pace_Score", 0.0)
                pm = row.iloc[0].get("Pace_Modifier", 1.0)
                vfm_pre = row.iloc[0].get("VFM_Pre_Pace", row.iloc[0].get("VFM", 0.0))
                vfm_post = row.iloc[0].get("VFM", 0.0)
                if ps > 0:
                    pace_info["pace_adjustments"].append(
                        {
                            "driver":        drv,
                            "pace_score":    round(ps, 1),
                            "pace_modifier": round(pm, 3),
                            "vfm_original":  round(vfm_pre, 2),
                            "vfm_adjusted":  round(vfm_post, 2),
                        }
                    )
        resp["optimization"]["fp2_info"] = pace_info

    return resp


def run_pending_tasks():
    with app.app_context():
        tasks = ScheduledOptimization.query.filter_by(status='pending').all()
        if not tasks:
            return
        settings = load_settings()
        for task in tasks:
            user = User.query.get(task.user_id)
            if not user:
                task.status = 'failed'
                task.result_json = json.dumps({'error': 'user not found'})
                continue
            try:
                config = json.loads(task.config_json)
                result = perform_optimization(config, user)
                html = render_template('email_results.html', opt=result['optimization'])
                send_email(user.email, 'F1 Optimisation Results', html, settings)
                task.status = 'completed'
                task.result_json = json.dumps(result)
            except Exception as e:
                traceback.print_exc()
                task.status = 'failed'
                task.result_json = json.dumps({'error': str(e)})
        db.session.commit()


def schedule_job():
    cron_expr = load_settings().get('cron_schedule', '0 18 * * 6')
    try:
        scheduler.remove_all_jobs()
    except Exception:
        pass
    try:
        trigger = CronTrigger.from_crontab(cron_expr, timezone=pytz.utc)
    except ValueError:
        print(f"Invalid cron expression: {cron_expr}, using default")
        trigger = CronTrigger.from_crontab('0 18 * * 6', timezone=pytz.utc)
    scheduler.add_job(run_pending_tasks, trigger, id='opt_job', replace_existing=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username and password:
            user = User.query.filter_by(provider="local", username=username).first()
            if user and check_password_hash(user.password_hash or "", password):
                admin_emails = [e.strip() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]
                new_admin = user.email in admin_emails
                if new_admin != user.admin:
                    user.admin = new_admin
                    db.session.commit()
                login_user(user)
                return redirect(url_for("index"))
            return render_template("login.html", message="Invalid credentials")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if not username or not email or not password:
            return render_template("register.html", message="All fields required")
        if User.query.filter_by(username=username).first():
            return render_template("register.html", message="Username taken")
        user = User(
            provider="local",
            provider_id=username,
            username=username,
            name=username,
            email=email,
        )
        user.password_hash = generate_password_hash(password)
        admin_emails = [e.strip() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]
        user.admin = email in admin_emails
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))
    return render_template("register.html")


@app.route("/login/<provider>")
def login_provider(provider):
    if provider not in ("google", "facebook", "github"):
        return redirect(url_for("login"))
    redirect_uri = url_for("authorize", provider=provider, _external=True)
    return oauth.create_client(provider).authorize_redirect(redirect_uri)


@app.route("/authorize/<provider>")
def authorize(provider):
    if provider not in ("google", "facebook", "github"):
        return redirect(url_for("login"))
    client = oauth.create_client(provider)
    token = client.authorize_access_token()
    if provider == "google":
        user_info = client.get("userinfo").json()
        provider_id = user_info.get("sub")
        email = user_info.get("email")
        name = user_info.get("name")
    elif provider == "facebook":
        user_info = client.get("me?fields=id,name,email").json()
        provider_id = user_info.get("id")
        email = user_info.get("email")
        name = user_info.get("name")
    else:
        user_info = client.get("user").json()
        provider_id = str(user_info.get("id"))
        name = user_info.get("name") or user_info.get("login")
        email = user_info.get("email")
        if not email:
            emails = client.get("user/emails").json()
            if emails:
                email = next((e["email"] for e in emails if e.get("primary")), emails[0]["email"])

    if not provider_id:
        return redirect(url_for("login"))

    user = User.query.filter_by(provider=provider, provider_id=provider_id).first()
    if not user:
        user = User(provider=provider, provider_id=provider_id, name=name, email=email)
        admin_emails = [e.strip() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()]
        user.admin = email in admin_emails
        db.session.add(user)
        db.session.commit()

    login_user(user)
    return redirect(url_for("index"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    cfg = None
    if current_user.config_json:
        try:
            cfg = json.loads(current_user.config_json)
        except Exception:
            cfg = None
    return render_template("index.html", user_config=cfg)


@app.route("/check_default_data")
@login_required
def check_default_data():
    default_info = load_default_data()
    return jsonify({"has_default": default_info is not None, "data": default_info})


@app.route("/check_driver_mapping")
@login_required
def check_driver_mapping():
    return jsonify({"exists": has_driver_mapping()})


@app.route("/upload", methods=["POST"])
@login_required
def upload_files():
    try:
        update_default = request.form.get("update_default", "false") == "true"
        session_id = request.form.get("session_id")
        if not session_id or update_default:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if update_default:
            target_folder = app.config["DEFAULT_DATA_FOLDER"]
        else:
            target_folder = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
            os.makedirs(target_folder, exist_ok=True)

        required_files = ["calendar.csv", "tracks.csv"]
        if not os.path.exists(os.path.join(target_folder, "driver_race_data.csv")):
            required_files.append("driver_race_data.csv")
        if not os.path.exists(os.path.join(target_folder, "constructor_race_data.csv")):
            required_files.append("constructor_race_data.csv")

        uploaded_files = []
        for field_name, file in request.files.items():
            if not file or not file.filename:
                continue

            filename = None
            for expected in required_files:
                if expected in field_name:
                    filename = expected
                    break
            if filename is None:
                filename = secure_filename(file.filename)

            dest_path = os.path.join(target_folder, filename)
            file.save(dest_path)
            uploaded_files.append(filename)

        missing = [f for f in required_files if f not in uploaded_files]
        if missing:
            return jsonify(
                {
                    "success": False,
                    "message": f"Missing required files: {', '.join(missing)}",
                }
            )

        folder_with_slash = target_folder if target_folder.endswith("/") else target_folder + "/"
        races_completed = get_races_completed(folder_with_slash)

        driver_df = pd.read_csv(os.path.join(folder_with_slash, "driver_race_data.csv"))
        constructor_df = pd.read_csv(os.path.join(folder_with_slash, "constructor_race_data.csv"))
        drivers_list = sorted(driver_df["Driver"].astype(str).unique().tolist())
        constructors_list = sorted(constructor_df["Constructor"].astype(str).unique().tolist())

        if not update_default:
            sessions[session_id] = {
                "folder": folder_with_slash,
                "races_completed": races_completed,
                "drivers": drivers_list,
                "constructors": constructors_list,
            }

        return jsonify(
            {
                "success": True,
                "session_id": "default" if update_default else session_id,
                "races_completed": races_completed,
                "drivers": drivers_list,
                "constructors": constructors_list,
                "updated_default": update_default,
                "has_driver_mapping": has_driver_mapping() if update_default else False,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": f"Error uploading files: {e}"})


@app.route("/upload_driver_mapping", methods=["POST"])
@login_required
def upload_driver_mapping():
    try:
        if "driver_mapping" not in request.files:
            return jsonify({"success": False, "message": "No driver mapping file provided"})

        file = request.files["driver_mapping"]
        if not file or not file.filename:
            return jsonify({"success": False, "message": "No file selected"})

        dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
        file.save(dest)

        mapping_df = pd.read_csv(dest)
        required_cols = ["driver_number", "driver_name", "team_name"]
        if not all(col in mapping_df.columns for col in required_cols):
            os.remove(dest)
            return jsonify(
                {
                    "success": False,
                    "message": f"driver_mapping.csv must contain columns: {', '.join(required_cols)}",
                }
            )
        if mapping_df.empty:
            os.remove(dest)
            return jsonify({"success": False, "message": "Driver mapping file is empty"})
        if not pd.api.types.is_numeric_dtype(mapping_df["driver_number"]):
            os.remove(dest)
            return jsonify({"success": False, "message": "driver_number must be numeric"})

        return jsonify(
            {
                "success": True,
                "message": f"Driver mapping uploaded ({len(mapping_df)} entries).",
                "driver_count": len(mapping_df),
            }
        )

    except Exception as e:
        dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
        if os.path.exists(dest):
            os.remove(dest)
        return jsonify({"success": False, "message": f"Error uploading driver mapping: {e}"})


@app.route("/optimize", methods=["POST"])
@login_required
def optimize():
    try:
        data = request.get_json() or {}
        result = perform_optimization(data, current_user)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"success": False, "message": str(e)})
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error during optimization"})


@app.route("/schedule_optimization", methods=["POST"])
@login_required
def schedule_optimization_route():
    data = request.get_json() or {}
    task = ScheduledOptimization(
        user_id=current_user.id,
        config_json=json.dumps(data),
        scheduled_for=datetime.utcnow(),
        status='pending',
    )
    db.session.add(task)
    db.session.commit()
    try:
        schedule_job()
    except Exception:
        pass
    return jsonify({"success": True})


@app.route("/statistics")
@login_required
def statistics():
    return render_template("statistics.html")


@app.route("/manage_data")
@login_required
def manage_data_page():
    if not current_user.admin:
        return redirect(url_for("index"))
    base = app.config["DEFAULT_DATA_FOLDER"]
    driver_path = os.path.join(base, "driver_race_data.csv")
    constructor_path = os.path.join(base, "constructor_race_data.csv")
    calendar_path = os.path.join(base, "calendar.csv")
    tracks_path = os.path.join(base, "tracks.csv")
    mapping_path = os.path.join(base, "driver_mapping.csv")

    driver_csv = ""
    constructor_csv = ""
    calendar_csv = ""
    tracks_csv = ""
    mapping_csv = ""
    if os.path.exists(driver_path):
        with open(driver_path, "r") as f:
            driver_csv = f.read()
    if os.path.exists(constructor_path):
        with open(constructor_path, "r") as f:
            constructor_csv = f.read()
    if os.path.exists(calendar_path):
        with open(calendar_path, "r") as f:
            calendar_csv = f.read()
    if os.path.exists(tracks_path):
        with open(tracks_path, "r") as f:
            tracks_csv = f.read()
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping_csv = f.read()

    settings = load_settings()

    message = request.args.get("message")
    return render_template(
        "manage_data.html",
        driver_csv=driver_csv,
        constructor_csv=constructor_csv,
        calendar_csv=calendar_csv,
        tracks_csv=tracks_csv,
        mapping_csv=mapping_csv,
        message=message,
        settings=settings,
    )


@app.route("/save_driver_data", methods=["POST"])
@login_required
def save_driver_data():
    csv_text = request.form.get("driver_data", "")
    dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_race_data.csv")
    try:
        pd.read_csv(io.StringIO(csv_text))
        with open(dest, "w") as f:
            f.write(csv_text)
        msg = "Driver data saved successfully."
    except Exception as e:
        msg = f"Failed to save driver data: {e}"
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/save_constructor_data", methods=["POST"])
@login_required
def save_constructor_data():
    csv_text = request.form.get("constructor_data", "")
    dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "constructor_race_data.csv")
    try:
        pd.read_csv(io.StringIO(csv_text))
        with open(dest, "w") as f:
            f.write(csv_text)
        msg = "Constructor data saved successfully."
    except Exception as e:
        msg = f"Failed to save constructor data: {e}"
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/save_calendar_data", methods=["POST"])
@login_required
def save_calendar_data():
    csv_text = request.form.get("calendar_data", "")
    dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "calendar.csv")
    try:
        pd.read_csv(io.StringIO(csv_text))
        with open(dest, "w") as f:
            f.write(csv_text)
        msg = "Calendar data saved successfully."
    except Exception as e:
        msg = f"Failed to save calendar data: {e}"
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/save_tracks_data", methods=["POST"])
@login_required
def save_tracks_data():
    csv_text = request.form.get("tracks_data", "")
    dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "tracks.csv")
    try:
        pd.read_csv(io.StringIO(csv_text))
        with open(dest, "w") as f:
            f.write(csv_text)
        msg = "Track data saved successfully."
    except Exception as e:
        msg = f"Failed to save track data: {e}"
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/save_mapping_data", methods=["POST"])
@login_required
def save_mapping_data():
    csv_text = request.form.get("mapping_data", "")
    dest = os.path.join(app.config["DEFAULT_DATA_FOLDER"], "driver_mapping.csv")
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        required_cols = ["driver_number", "driver_name", "team_name"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"driver_mapping.csv must contain columns: {', '.join(required_cols)}"
            )
        if df.empty:
            raise ValueError("Driver mapping file is empty")
        if not pd.api.types.is_numeric_dtype(df["driver_number"]):
            raise ValueError("driver_number must be numeric")
        df.to_csv(dest, index=False)
        msg = "Driver mapping saved successfully."
    except Exception as e:
        msg = f"Failed to save driver mapping: {e}"
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/save_settings", methods=["POST"])
@login_required
def save_settings_route():
    data = {
        "outlier_stddev_factor": request.form.get("outlier_stddev_factor", type=float),
        "trend_slope_threshold": request.form.get("trend_slope_threshold", type=float),
        "recent_races_fraction": request.form.get("recent_races_fraction", type=float),
        "long_term_weight": request.form.get("long_term_weight", type=float),
        "interaction_weight": request.form.get("interaction_weight", type=float),
        "top_n_candidates": request.form.get("top_n_candidates", type=int),
        "use_ilp": bool(request.form.get("use_ilp")),
        "cron_schedule": request.form.get("cron_schedule", "0 18 * * 6"),
        "smtp_host": request.form.get("smtp_host", ""),
        "smtp_port": request.form.get("smtp_port", type=int, default=587),
        "smtp_username": request.form.get("smtp_username", ""),
        "smtp_password": request.form.get("smtp_password", ""),
        "smtp_tls": bool(request.form.get("smtp_tls")),
        "smtp_from": request.form.get("smtp_from", ""),
    }
    success = save_settings(data)
    if success:
        try:
            schedule_job()
        except Exception:
            pass
    msg = "Settings saved." if success else "Failed to save settings."
    return redirect(url_for("manage_data_page", message=msg))


@app.route("/send_test_email", methods=["POST"])
@login_required
def send_test_email_route():
    if not current_user.admin:
        return jsonify({"success": False, "message": "Unauthorized"})
    settings = load_settings()
    html = "<p>This is a test email from the F1 Fantasy optimiser.</p>"
    ok = send_email(current_user.email, "SMTP Test", html, settings)
    if ok:
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Failed to send email"})


@app.route("/api/statistics")
@login_required
def get_statistics():
    try:
        data_folder = get_data_folder("default")
        if not data_folder:
            return jsonify({"success": False, "message": "No data; upload files first."})

        driver_race_df = pd.read_csv(os.path.join(data_folder, "driver_race_data.csv"))
        constructor_race_df = pd.read_csv(os.path.join(data_folder, "constructor_race_data.csv"))
        calendar_df = pd.read_csv(os.path.join(data_folder, "calendar.csv"))
        tracks_df = pd.read_csv(os.path.join(data_folder, "tracks.csv"))

        driver_aff_path = os.path.join(data_folder, "driver_affinity.csv")
        constructor_aff_path = os.path.join(data_folder, "constructor_affinity.csv")
        driver_char_aff = os.path.join(data_folder, "driver_characteristic_affinities.csv")
        constructor_char_aff = os.path.join(data_folder, "constructor_characteristic_affinities.csv")

        if not all(os.path.exists(p) for p in [driver_aff_path, constructor_aff_path, driver_char_aff, constructor_char_aff]):
            cfg = {
                "base_path":         data_folder,
                "races_completed":   get_races_completed(data_folder),
                "weighting_scheme":  "trend_based",
                "use_fp2_pace":      False,
            }
            vfm_calc = F1VFMCalculator(cfg)
            vfm_calc.run()
            aff_calc = F1TrackAffinityCalculator(cfg)
            aff_calc.run()

        driver_vfm_df = pd.read_csv(os.path.join(data_folder, "driver_vfm.csv"))
        constructor_vfm_df = pd.read_csv(os.path.join(data_folder, "constructor_vfm.csv"))
        driver_aff_df = pd.read_csv(driver_aff_path)
        constructor_aff_df = pd.read_csv(constructor_aff_path)
        driver_char_df = pd.read_csv(driver_char_aff, index_col=0)
        constructor_char_df = pd.read_csv(constructor_char_aff, index_col=0)

        driver_stats = process_driver_statistics(
            driver_race_df, driver_vfm_df, driver_aff_df, driver_char_df, calendar_df, tracks_df
        )
        constructor_stats = process_constructor_statistics(
            constructor_race_df, constructor_vfm_df, constructor_aff_df, constructor_char_df, calendar_df, tracks_df
        )

        return jsonify(
            {
                "success": True,
                "drivers": driver_stats,
                "constructors": constructor_stats,
                "races_completed": get_races_completed(data_folder),
            }
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error calculating statistics"})


def process_driver_statistics(
    driver_race_df,
    driver_vfm_df,
    driver_aff_df,
    driver_char_df,
    calendar_df,
    tracks_df,
):
    stats_list = []
    race_cols = [c for c in driver_race_df.columns if c.startswith("Race")]

    for _, drv_row in driver_vfm_df.iterrows():
        name = str(drv_row["Driver"])
        race_data = driver_race_df[driver_race_df["Driver"] == name]
        if race_data.empty:
            continue

        points_arr = race_data[race_cols].values.flatten()
        valid_pts = [float(p) for p in points_arr if not np.isnan(p)]

        s = {
            "name":            name,
            "team":            str(drv_row.get("Team", "Unknown")),
            "cost":            str(drv_row["Cost"]),
            "cost_value":      float(re.sub(r"[^\d.]", "", str(drv_row["Cost"]))),
            "vfm":             float(drv_row["VFM"]),
            "trend":           str(drv_row.get("Performance_Trend", "Unknown")),
            "avg_points":      float(np.mean(valid_pts)) if valid_pts else 0.0,
            "total_points":    float(np.sum(valid_pts)) if valid_pts else 0.0,
            "races_completed": len(valid_pts),
            "consistency":     float(np.std(valid_pts)) if len(valid_pts) > 1 else 0.0,
            "best_race":       float(np.max(valid_pts)) if valid_pts else 0.0,
            "worst_race":      float(np.min(valid_pts)) if valid_pts else 0.0,
        }

        if len(valid_pts) >= 3:
            last3 = valid_pts[-3:]
            s["recent_form"] = float(np.mean(last3) - s["avg_points"])
        else:
            s["recent_form"] = 0.0

        if name in driver_char_df.index:
            char_vals = driver_char_df.loc[name]
            s["char_affinities"] = {
                "Corners":      float(char_vals.get("Corners", 0.0)),
                "Length":       float(char_vals.get("Length (km)", 0.0)),
                "Overtaking":   float(char_vals.get("Overtaking Opportunities_encoded", 0.0)),
                "Speed":        float(char_vals.get("Track Speed_encoded", 0.0)),
                "Temperature":  float(char_vals.get("Expected Temperatures_encoded", 0.0)),
            }
        else:
            s["char_affinities"] = {
                "Corners":      0.0,
                "Length":       0.0,
                "Overtaking":   0.0,
                "Speed":        0.0,
                "Temperature":  0.0,
            }

        perf_list = []
        for _, cal_row in calendar_df.iterrows():
            race_name = cal_row["Race"]
            if race_name in race_cols:
                idx = race_cols.index(race_name)
                if idx < len(points_arr) and not np.isnan(points_arr[idx]):
                    circuit = str(cal_row["Circuit"])
                    affinity_col = f"{name}_affinity"
                    aff = 0.0
                    if affinity_col in driver_aff_df.columns:
                        circ_row = driver_aff_df[driver_aff_df["Circuit"] == circuit]
                        if not circ_row.empty:
                            aff = float(circ_row.iloc[0][affinity_col])
                    perf_list.append({"circuit": circuit, "points": float(points_arr[idx]), "affinity": aff})

        perf_list.sort(key=lambda x: x["points"], reverse=True)
        s["best_tracks"] = perf_list[:3] if len(perf_list) >= 3 else perf_list
        s["worst_tracks"] = perf_list[-3:] if len(perf_list) >= 3 else []

        upc = []
        completed = len([r for r in race_cols if r in calendar_df["Race"].values])
        for i in range(completed + 1, min(completed + 4, len(race_cols) + 1)):
            rname = f"Race{i}"
            row = calendar_df[calendar_df["Race"] == rname]
            if not row.empty:
                circuit = str(row.iloc[0]["Circuit"])
                aff_col = f"{name}_affinity"
                aff_val = 0.0
                if aff_col in driver_aff_df.columns:
                    circ_row = driver_aff_df[driver_aff_df["Circuit"] == circuit]
                    if not circ_row.empty:
                        aff_val = float(circ_row.iloc[0][aff_col])
                upc.append({"race": rname, "circuit": circuit, "affinity": aff_val})

        s["upcoming_races"] = upc
        stats_list.append(s)

    stats_list.sort(key=lambda x: x["vfm"], reverse=True)
    for rank, entry in enumerate(stats_list, start=1):
        entry["vfm_rank"] = rank

    return stats_list


def process_constructor_statistics(
    constructor_race_df,
    constructor_vfm_df,
    constructor_aff_df,
    constructor_char_df,
    calendar_df,
    tracks_df,
):
    stats_list = []
    race_cols = [c for c in constructor_race_df.columns if c.startswith("Race")]

    for _, const_row in constructor_vfm_df.iterrows():
        name = str(const_row["Constructor"])
        race_data = constructor_race_df[constructor_race_df["Constructor"] == name]
        if race_data.empty:
            continue

        points_arr = race_data[race_cols].values.flatten()
        valid_pts = [float(p) for p in points_arr if not np.isnan(p)]

        s = {
            "name":            name,
            "cost":            str(const_row["Cost"]),
            "cost_value":      float(re.sub(r"[^\d.]", "", str(const_row["Cost"]))),
            "vfm":             float(const_row["VFM"]),
            "trend":           str(const_row.get("Performance_Trend", "Unknown")),
            "avg_points":      float(np.mean(valid_pts)) if valid_pts else 0.0,
            "total_points":    float(np.sum(valid_pts)) if valid_pts else 0.0,
            "races_completed": len(valid_pts),
            "consistency":     float(np.std(valid_pts)) if len(valid_pts) > 1 else 0.0,
            "best_race":       float(np.max(valid_pts)) if valid_pts else 0.0,
            "worst_race":      float(np.min(valid_pts)) if valid_pts else 0.0,
        }

        if len(valid_pts) >= 3:
            last3 = valid_pts[-3:]
            s["recent_form"] = float(np.mean(last3) - s["avg_points"])
        else:
            s["recent_form"] = 0.0

        if name in constructor_char_df.index:
            char_vals = constructor_char_df.loc[name]
            s["char_affinities"] = {
                "Corners":      float(char_vals.get("Corners", 0.0)),
                "Length":       float(char_vals.get("Length (km)", 0.0)),
                "Overtaking":   float(char_vals.get("Overtaking Opportunities_encoded", 0.0)),
                "Speed":        float(char_vals.get("Track Speed_encoded", 0.0)),
                "Temperature":  float(char_vals.get("Expected Temperatures_encoded", 0.0)),
            }
        else:
            s["char_affinities"] = {
                "Corners":      0.0,
                "Length":       0.0,
                "Overtaking":   0.0,
                "Speed":        0.0,
                "Temperature":  0.0,
            }

        perf_list = []
        for _, cal_row in calendar_df.iterrows():
            race_name = cal_row["Race"]
            if race_name in race_cols:
                idx = race_cols.index(race_name)
                if idx < len(points_arr) and not np.isnan(points_arr[idx]):
                    circuit = str(cal_row["Circuit"])
                    aff_col = f"{name}_affinity"
                    aff_val = 0.0
                    if aff_col in constructor_aff_df.columns:
                        circ_row = constructor_aff_df[constructor_aff_df["Circuit"] == circuit]
                        if not circ_row.empty:
                            aff_val = float(circ_row.iloc[0][aff_col])
                    perf_list.append({"circuit": circuit, "points": float(points_arr[idx]), "affinity": aff_val})

        perf_list.sort(key=lambda x: x["points"], reverse=True)
        s["best_tracks"] = perf_list[:3] if len(perf_list) >= 3 else perf_list
        s["worst_tracks"] = perf_list[-3:] if len(perf_list) >= 3 else []

        upc = []
        completed = len([r for r in race_cols if r in calendar_df["Race"].values])
        for i in range(completed + 1, min(completed + 4, len(race_cols) + 1)):
            rname = f"Race{i}"
            row = calendar_df[calendar_df["Race"] == rname]
            if not row.empty:
                circuit = str(row.iloc[0]["Circuit"])
                aff_col = f"{name}_affinity"
                aff_val = 0.0
                if aff_col in constructor_aff_df.columns:
                    circ_row = constructor_aff_df[constructor_aff_df["Circuit"] == circuit]
                    if not circ_row.empty:
                        aff_val = float(circ_row.iloc[0][aff_col])
                upc.append({"race": rname, "circuit": circuit, "affinity": aff_val})

        s["upcoming_races"] = upc
        stats_list.append(s)

    stats_list.sort(key=lambda x: x["vfm"], reverse=True)
    for rank, entry in enumerate(stats_list, start=1):
        entry["vfm_rank"] = rank

    return stats_list


@app.route("/api/export_statistics")
@login_required
def export_statistics():
    try:
        stats_resp = get_statistics()
        stats_data = stats_resp.get_json()
        if not stats_data.get("success", False):
            return stats_resp

        driver_rows = []
        for drv in stats_data["drivers"]:
            driver_rows.append(
                {
                    "Driver":      drv["name"],
                    "Team":        drv["team"],
                    "Cost":        drv["cost"],
                    "VFM":         drv["vfm"],
                    "VFM_Rank":    drv["vfm_rank"],
                    "Avg_Points":  round(drv["avg_points"], 2),
                    "Total_Points":drv["total_points"],
                    "Consistency": round(drv["consistency"], 2),
                    "Recent_Form": round(drv["recent_form"], 2),
                    "Trend":       drv["trend"],
                }
            )
        df_drv = pd.DataFrame(driver_rows)

        const_rows = []
        for const in stats_data["constructors"]:
            const_rows.append(
                {
                    "Constructor": const["name"],
                    "Cost":        const["cost"],
                    "VFM":         const["vfm"],
                    "VFM_Rank":    const["vfm_rank"],
                    "Avg_Points":  round(const["avg_points"], 2),
                    "Total_Points":const["total_points"],
                    "Consistency": round(const["consistency"], 2),
                    "Recent_Form": round(const["recent_form"], 2),
                    "Trend":       const["trend"],
                }
            )
        df_const = pd.DataFrame(const_rows)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(app.config["RESULTS_FOLDER"], f"statistics_{ts}.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_drv.to_excel(writer, sheet_name="Drivers", index=False)
            df_const.to_excel(writer, sheet_name="Constructors", index=False)

        return send_file(
            out_path,
            as_attachment=True,
            download_name=f"f1_statistics_{ts}.xlsx",
        )
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Error exporting statistics"})
with app.app_context():
    db.create_all()
    schedule_job()
    if not scheduler.running:
        scheduler.start()




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
