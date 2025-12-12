import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
import sys
import base64
import time
import os
import io
import csv
import threading
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import requests
import shutil

# ==========================
#    CONFIG & CONSTANTES
# ==========================
st.set_page_config(
    page_title="AI Librarian – Recommandations de livres",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

KAGGLE_COMP = "the-lazy-librarian-build-an-ai-to-do-your-job"
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent  # dossier avant ../
SUBMISSION_FILE = ROOT_DIR / "submission.csv"      # nouveau submission.csv généré par le modèle
ITEMS_FILE = ROOT_DIR / "books_enriched_FINAL.csv"
INTERACTIONS_FILE = ROOT_DIR / "interactions_train.csv"
MODEL_SCRIPT = ROOT_DIR / "create_submission.py"   # script du modèle
GIF_PATH = BASE_DIR / "Mapping for machine learning.gif"
LOCAL_KAGGLE = BASE_DIR / "kaggle.json"
fast_script = ROOT_DIR / "generate_submission_fast.py"

def setup_kaggle_credentials():
    """
    Copie le kaggle.json du dossier du projet → ~/.kaggle/kaggle.json
    pour permettre à Kaggle CLI + KaggleApi de fonctionner sur Streamlit Cloud.
    """
    if not LOCAL_KAGGLE.exists():
        print("⚠️ Aucun kaggle.json trouvé dans le projet")
        return

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_target = kaggle_dir / "kaggle.json"

    shutil.copy(LOCAL_KAGGLE, kaggle_target)
    os.chmod(kaggle_target, 0o600)

    # Pour KaggleApi Python
    with open(LOCAL_KAGGLE) as f:
        data = json.load(f)

    os.environ["KAGGLE_USERNAME"] = data["username"]
    os.environ["KAGGLE_KEY"] = data["key"]

    print("✅ Kaggle credentials loaded depuis kaggle.json du projet")

# appelle immédiatement :
setup_kaggle_credentials()
# ==========================
#         STYLES
# ==========================
CUSTOM_CSS = """
<style>

    /* Fond général */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #0b1120 40%, #020617 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    /* Titres principaux */
    h1, h2, h3 {
        color: #e5e7eb !important;
    }

    /* Étoiles filantes réalistes */
    .shooting-star {
        position: fixed;
        width: 3px;
        height: 3px;
        background: white;
        border-radius: 50%;
        box-shadow:
            0 0 6px rgba(255,255,255,0.9),
            0 0 12px rgba(147,197,253,0.8);
        opacity: 0;
        pointer-events: none;
        z-index: 50;
        animation: shootingStarMove linear infinite, shootingStarTail linear infinite;
    }

    /* Trajectoire diagonal + wobble + fade */
    @keyframes shootingStarMove {
        0% {
            opacity: 0;
            transform: translate(0, 0) scale(0.6);
        }
        5% {
            opacity: 1;
        }
        50% {
            transform: translate(var(--dx), var(--dy)) scale(1.05) rotate(var(--angle));
        }
        100% {
            opacity: 0;
            transform: translate(calc(var(--dx) * 1.5), calc(var(--dy) * 1.5)) scale(0.9);
        }
    }

    /* Tracé lumineux NON DROIT */
    @keyframes shootingStarTail {
        0% {
            box-shadow:
                -10px -5px 20px rgba(147,197,253,0.8),
                -20px -10px 30px rgba(59,130,246,0.6);
        }
        50% {
            box-shadow:
                -15px -8px 25px rgba(147,197,253,0.9),
                -35px -18px 40px rgba(168,85,247,0.7);
        }
        100% {
            box-shadow:
                -30px -10px 40px rgba(147,197,253,0.0),
                -60px -25px 60px rgba(168,85,247,0.0);
        }
    }

    /* ========== MODEL CARDS (HOME SCORE SECTION) ========== */

    .model-section-title {
        margin-top: 1.5rem;
        font-size: 0.95rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        text-align: center;
    }

    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-top: 1rem;
    }

    .model-card {
        background: rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 10px 14px;
        border-radius: 14px;
        text-align: center;
        font-size: 0.80rem;
        color: #cbd5e1;
        backdrop-filter: blur(10px);
        transition: all 0.25s ease;
        box-shadow: 0 0 18px rgba(59,130,246,0.15);
    }

    .model-card:hover {
        transform: translateY(-4px);
        border-color: #60a5fa;
        box-shadow: 0 0 22px rgba(59,130,246,0.35);
    }

    .model-card-title {
        font-weight: 600;
        margin-bottom: 4px;
        color: #e2e8f0;
    }

    .model-tag {
        font-size: 0.65rem;
        padding: 3px 8px;
        border-radius: 8px;
        background: rgba(59,130,246,0.15);
        color: #93c5fd;
        display: inline-block;
        margin-top: 4px;
    }

    /* Cartes de livres – forme 3D de livre animé */
    @keyframes bookFloat {
        0%, 100% { transform: perspective(1000px) rotateY(-5deg) translateY(0px); }
        50% { transform: perspective(1000px) rotateY(-5deg) translateY(-8px); }
    }

    @keyframes bookGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(59,130,246,0.3), inset 0 0 20px rgba(59,130,246,0.1); }
        50% { box-shadow: 0 25px 70px rgba(168,85,247,0.4), inset 0 0 25px rgba(168,85,247,0.15); }
    }

    @keyframes spineGlow {
        0%, 100% { box-shadow: -2px 0 15px rgba(59,130,246,0.6); }
        50% { box-shadow: -2px 0 20px rgba(168,85,247,0.8); }
    }

    .book-card {
        position: relative;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 0 8px 8px 0;
        padding: 20px 24px 20px 32px;
        margin: 20px auto;
        max-width: 600px;
        border: 1px solid #334155;
        transform-style: preserve-3d;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: bookFloat 4s ease-in-out infinite, bookGlow 3s ease-in-out infinite;
        cursor: pointer;
    }

    .book-card:hover {
        transform: perspective(1000px) rotateY(2deg) translateY(-12px) scale(1.02);
        border-color: #60a5fa;
    }

    /* Tranche du livre (spine) */
    .book-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 12px;
        background: linear-gradient(180deg, 
            #1d4ed8 0%, 
            #3b82f6 25%, 
            #a855f7 50%, 
            #8b5cf6 75%, 
            #6366f1 100%);
        border-radius: 8px 0 0 8px;
        animation: spineGlow 2s ease-in-out infinite;
    }

    /* Pages du livre (effet 3D) */
    .book-card::after {
        content: "";
        position: absolute;
        right: 0;
        top: 4px;
        bottom: 4px;
        width: 8px;
        background: repeating-linear-gradient(
            0deg,
            #1e293b,
            #1e293b 2px,
            #0f172a 2px,
            #0f172a 4px
        );
        border-radius: 0 4px 4px 0;
        opacity: 0.6;
    }

    .book-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 6px;
        color: #f1f5f9;
        text-shadow: 0 2px 8px rgba(59,130,246,0.3);
        letter-spacing: 0.02em;
    }

    .book-meta {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 8px;
        font-weight: 500;
    }

    .book-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(168,85,247,0.2));
        color: #bfdbfe;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(59,130,246,0.3);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Card "stat" */
    .stat-card {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        border-radius: 18px;
        padding: 18px 20px;
        border: 1px solid rgba(148,163,184,0.3);
        box-shadow: 0 18px 40px rgba(15,23,42,0.7);
    }

    .hero-container {
        text-align: center;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: linear-gradient(120deg, #60a5fa, #a855f7, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #9ca3af;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.5);
        margin-bottom: 0.75rem;
    }

    /* Animations pour le score */
    @keyframes scoreFill {
        0% { 
            background: conic-gradient(from 220deg, rgba(34,197,94,0.3) 0deg, rgba(31,41,55,0.9) 0deg);
        }
        100% {
            background: conic-gradient(from 220deg, #22c55e 0deg, #22c55e 61.2deg, rgba(31,41,55,0.9) 61.2deg);
        }
    }

    @keyframes scoreGlow {
        0%, 100% { 
            box-shadow: 0 0 25px rgba(34,197,94,0.4), 0 0 50px rgba(34,197,94,0.2); 
        }
        50% { 
            box-shadow: 0 0 40px rgba(34,197,94,0.8), 0 0 70px rgba(34,197,94,0.4); 
        }
    }

    @keyframes scoreSpinner {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes scorePulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
    }

    .score-ring {
        width: 140px;
        height: 140px;
        border-radius: 999px;
        margin: 0 auto;
        background: conic-gradient(from 220deg, rgba(34,197,94,0.3) 0deg, rgba(31,41,55,0.9) 0deg);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 25px rgba(34,197,94,0.4);
        animation: scoreFill 2s ease-out 0.3s forwards, scoreGlow 2s ease-in-out 0.3s infinite;
        position: relative;
    }

    .score-ring.loading {
        background: conic-gradient(from 0deg, #3b82f6 0deg, #3b82f6 90deg, rgba(31,41,55,0.9) 90deg);
        animation: scoreSpinner 1s linear infinite;
    }

    .score-ring-inner {
        width: 96px;
        height: 96px;
        border-radius: 999px;
        background: radial-gradient(circle, #0f172a 0%, #020617 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(148,163,184,0.5);
        animation: scorePulse 2s ease-in-out 0.3s 1;
    }

    .score-ring-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #22c55e, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .score-ring-label {
        font-size: 0.65rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }

    .cta-row {
        margin-top: 2.0rem;
    }

    /* Loader orb IA */
    @keyframes orbitSpin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes orbPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 18px rgba(59,130,246,0.6); }
        50% { transform: scale(1.1); box-shadow: 0 0 26px rgba(168,85,247,0.8); }
    }

    .loader-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .loader-orb {
        position: relative;
        width: 70px;
        height: 70px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 20%, #bae6fd, #1d4ed8 40%, #020617 70%);
        animation: orbPulse 1.8s ease-in-out infinite;
    }

    .loader-orb::before,
    .loader-orb::after {
        content: "";
        position: absolute;
        inset: -12px;
        border-radius: 999px;
        border: 1px dashed rgba(148,163,184,0.6);
        animation: orbitSpin 10s linear infinite;
    }

    .loader-orb::after {
        inset: -22px;
        border-style: solid;
        border-color: rgba(59,130,246,0.5) transparent rgba(168,85,247,0.6) transparent;
        animation-duration: 16s;
    }

    /* Boutons LED / IA */
    .stButton > button {
        border-radius: 999px;
        padding: 0.9rem 1.8rem;
        border: 1px solid rgba(148,163,184,0.6);
        background: radial-gradient(circle at 10% 0, #1d4ed8 0, #0f172a 50%, #020617 100%);
        color: #e5e7eb;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 0 30px rgba(59,130,246,0.45);
        text-shadow: 0 0 10px rgba(129,140,248,0.8);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.07);
        border-color: #e5e7eb;
        box-shadow: 0 0 35px rgba(129,140,248,0.8);
        transform: translateY(-2px);
    }

    /* Champs de saisie, select, etc. */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox select {
        background: #020617 !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
        border-radius: 12px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
    }

    /* ========== NEW: LOGIN PAGE ANIMATIONS ========== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes glowPulse {
        0%, 100% {
            text-shadow: 0 0 20px rgba(59,130,246,0.6), 0 0 40px rgba(168,85,247,0.4);
        }
        50% {
            text-shadow: 0 0 30px rgba(59,130,246,0.8), 0 0 60px rgba(168,85,247,0.6);
        }
    }

    @keyframes particleFloat {
        0%, 100% {
            transform: translateY(0) translateX(0);
            opacity: 0.3;
        }
        50% {
            transform: translateY(-20px) translateX(10px);
            opacity: 0.6;
        }
    }

    @keyframes slideInFromLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Login container plus compact, collé en haut */
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        padding-top: 1.25rem;
        padding-bottom: 0.75rem;
        animation: fadeInUp 0.8s ease-out;
    }

    .login-card {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 20px;
        padding: 2rem 2.25rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55), 0 0 70px rgba(59, 130, 246, 0.25);
        max-width: 420px;
        width: 100%;
        animation: fadeInUp 1s ease-out 0.1s both;
    }

    .login-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #60a5fa, #a855f7, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.4rem;
        animation: glowPulse 3s ease-in-out infinite;
    }

    .login-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
    }

    .particle {
        position: fixed;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, #60a5fa, transparent);
        border-radius: 50%;
        animation: particleFloat 4s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    /* Étoiles brillantes en arrière-plan */
    @keyframes twinkle {
        0%, 100% { 
            opacity: 0.3;
            transform: scale(1);
        }
        50% { 
            opacity: 1;
            transform: scale(1.2);
        }
    }

    .star {
        position: fixed;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        box-shadow: 0 0 3px rgba(255,255,255,0.8), 0 0 6px rgba(255,255,255,0.4);
        animation: twinkle 3s ease-in-out infinite;
    }

    /* Fusées naturelles - trajectoire diagonale simple */
    @keyframes shootingStarNatural {
        0% {
            transform: translate3d(-100px, -100px, 0) rotate(-45deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 0.8;
        }
        100% {
            transform: translate3d(calc(100vw + 100px), calc(100vh + 100px), 0) rotate(-45deg);
            opacity: 0;
        }
    }
    @keyframes rocketLoop {
        0% {
            transform: translate3d(-120px, -80px, 0) rotate(-35deg);
            opacity: 0;
        }
        10% { opacity: 1; }
        50% {
            transform: translate3d(40vw, 30vh, 0) rotate(-40deg);
            opacity: 0.9;
        }
        100% {
            transform: translate3d(110vw, 110vh, 0) rotate(-45deg);
            opacity: 0;
        }
    }

    @keyframes rocketZigzag {
        0% {
            transform: translate3d(-120px, -60px, 0) rotate(-30deg);
            opacity: 0;
        }
        15% { opacity: 1; }
        40% {
            transform: translate3d(30vw, 10vh, 0) rotate(-40deg);
        }
        70% {
            transform: translate3d(60vw, 40vh, 0) rotate(-50deg);
        }
        100% {
            transform: translate3d(110vw, 100vh, 0) rotate(-55deg);
            opacity: 0;
        }
    }

    @keyframes rocketSpiral {
        0% {
            transform: translate3d(-140px, -100px, 0) rotate(-20deg) scale(0.9);
            opacity: 0;
        }
        20% { opacity: 1; }
        50% {
            transform: translate3d(50vw, 30vh, 0) rotate(-90deg) scale(1.05);
        }
        100% {
            transform: translate3d(115vw, 110vh, 0) rotate(-140deg) scale(0.95);
            opacity: 0;
        }
    }

    @keyframes rocketWave {
        0% {
            transform: translate3d(-140px, -70px, 0) rotate(-30deg);
            opacity: 0;
        }
        20% { opacity: 1; }
        50% {
            transform: translate3d(45vw, 15vh, 0) rotate(-40deg) translateY(-20px);
        }
        80% {
            transform: translate3d(80vw, 60vh, 0) rotate(-50deg) translateY(15px);
        }
        100% {
            transform: translate3d(115vw, 105vh, 0) rotate(-55deg);
            opacity: 0;
        }
    }

    @keyframes rocketCrazy {
        0% {
            transform: translate3d(-150px, -120px, 0) rotate(-25deg);
            opacity: 0;
        }
        15% { opacity: 1; }
        35% {
            transform: translate3d(25vw, 5vh, 0) rotate(-40deg);
        }
        60% {
            transform: translate3d(55vw, 35vh, 0) rotate(-70deg);
        }
        85% {
            transform: translate3d(85vw, 70vh, 0) rotate(-50deg);
        }
        100% {
            transform: translate3d(120vw, 120vh, 0) rotate(-60deg);
            opacity: 0;
        }
    }
    .shooting-star {
        position: fixed;
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, 
            rgba(255,255,255,0) 0%, 
            rgba(255,255,255,0.1) 10%,
            rgba(147,197,253,0.8) 50%, 
            rgba(255,255,255,1) 70%,
            rgba(255,255,255,0) 100%);
        border-radius: 999px;
        pointer-events: none;
        z-index: 1;
        filter: drop-shadow(0 0 6px rgba(147,197,253,0.8)) blur(0.5px);
        animation: shootingStarNatural linear infinite;
        opacity: 0;
    }

    .shooting-star::before {
        content: "";
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 4px;
        background: white;
        border-radius: 50%;
        box-shadow: 0 0 10px rgba(255,255,255,0.9), 0 0 20px rgba(147,197,253,0.6);
    }

    /* ========== SIDEBAR STYLING ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
    }

    .sidebar-user-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
        animation: slideInFromLeft 0.6s ease-out;
    }

    .sidebar-user-id {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        color: transparent;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .nav-item:hover {
        background: rgba(59, 130, 246, 0.2);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateX(5px);
    }

    .nav-item-active {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(168, 85, 247, 0.3));
        border-color: rgba(59, 130, 246, 0.6);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }

    /* Welcome message animation */
    .welcome-message {
        text-align: center;
        padding: 2rem 0;
        animation: fadeInUp 1s ease-out;
    }

    .welcome-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a855f7, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 1rem;
        animation: glowPulse 3s ease-in-out infinite;
    }

    .welcome-subtitle {
        font-size: 1.2rem;
        color: #cbd5e1;
        margin-bottom: 0.5rem;
    }

    /* Animation de flottement du GIF */
    @keyframes floatGif {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    /* Animation de montée depuis le bas au chargement */
    @keyframes slideUpFromBottom {
        0% {
            transform: translateY(100vh);
            opacity: 0;
        }
        60% {
            opacity: 1;
        }
        100% {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .floating-gif {
        animation: slideUpFromBottom 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) 0s 1 normal both,
                floatGif 3.5s ease-in-out 1.2s infinite;
        display: flex;
        justify-content: center;
        margin-top: 4px;
        margin-bottom: 2px;
    }

    /* ========== TABLEAUX JOLIS ========== */

    .nice-table-container {
        max-height: 420px;
        overflow-y: auto;
        border-radius: 16px;
        border: 1px solid rgba(148,163,184,0.35);
        background: radial-gradient(circle at top left, #020617, #020617 40%, #020617);
        box-shadow: 0 14px 40px rgba(15,23,42,0.9);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .nice-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        color: #e5e7eb;
    }

    .nice-table thead {
        background: linear-gradient(90deg, rgba(30,64,175,0.8), rgba(76,29,149,0.8));
        position: sticky;
        top: 0;
        z-index: 1;
    }

    .nice-table thead th {
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.7rem;
    }

    .nice-table tbody tr:nth-child(even) {
        background: rgba(15,23,42,0.85);
    }

    .nice-table tbody tr:nth-child(odd) {
        background: rgba(15,23,42,0.6);
    }

    .nice-table tbody tr:hover {
        background: rgba(37,99,235,0.35);
    }

    .nice-table tbody td {
        padding: 8px 12px;
        border-top: 1px solid rgba(30,64,175,0.4);
        font-size: 0.82rem;
    }
    /* ==========================
    HEADER / FOOTER STREAMLIT
    ========================== */

    /* On garde le header pour pouvoir ouvrir/fermer la sidebar,
    mais on le rend discret pour rester dans le thème custom */
    header[data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }

    /* On cache le menu ... de Streamlit */
    #MainMenu {
        display: none;
    }

    /* On cache le footer "Made with Streamlit" */
    footer {
        display: none;
    }

    /* Laisse le scroll normal, sinon certaines parties (login, fusées) peuvent être coupées */
    html, body {
        overflow-y: auto;
    }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
def render_pretty_table(df: pd.DataFrame, max_height: int = 420):
    """Affiche un tableau stylé avec CSS custom."""
    if df is None or df.empty:
        st.info("Aucune donnée à afficher.")
        return
    
    html_table = df.to_html(
        index=False,
        classes="nice-table",
        escape=False
    )
    st.markdown(
        f"""
        <div class="nice-table-container" style="max-height:{max_height}px;">
            {html_table}
        </div>
        """,
        unsafe_allow_html=True
    )

def background_kaggle_submit(file_path, message, callback_status):
    """
    Exécute la commande Kaggle en tâche de fond.
    callback_status est un dict dans session_state pour stocker l'état du submit.
    """
    import subprocess

    callback_status["status"] = "running"

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "the-lazy-librarian-build-an-ai-to-do-your-job",
        "-f", file_path,
        "-m", message
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        output = proc.communicate()[0]
        callback_status["output"] = output

        if proc.returncode == 0:
            callback_status["status"] = "success"
        else:
            callback_status["status"] = "failed"

    except Exception as e:
        callback_status["status"] = "error"
        callback_status["output"] = str(e)

def kaggle_get_leaderboard():
    """
    Récupère le leaderboard Kaggle pour la compétition.
    Utilise : kaggle competitions leaderboard -c … --show --csv
    Retourne : DataFrame ou None
    """
    cmd = [
        "kaggle", "competitions", "leaderboard",
        "-c", KAGGLE_COMP,
        "--show",
        "--csv"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        st.error("❌ La commande `kaggle` n'est pas disponible sur ce serveur (leaderboard).")
        return None
    except subprocess.CalledProcessError as e:
        st.error("❌ Erreur lors de la récupération du leaderboard Kaggle.")
        st.code(e.stderr or e.stdout, language="bash")
        return None

    # Si aucune sortie CSV
    if not result.stdout.strip():
        st.info("Aucun leaderboard trouvé pour cette compétition.")
        return None

    # Convertir CSV → DataFrame
    try:
        df = pd.read_csv(io.StringIO(result.stdout))
        return df
    except Exception as e:
        st.error(f"Impossible de parser le leaderboard Kaggle : {e}")
        return None


def kaggle_submit_submission(submission_path: Path, message: str):
    """
    Envoie `submission_path` sur Kaggle avec un message et retourne True/False.
    """
    if not submission_path.exists():
        st.error(f"❌ Fichier introuvable pour la soumission : {submission_path}")
        return False

    cmd = [
        "kaggle", "competitions", "submit",
        "-c", KAGGLE_COMP,
        "-f", str(submission_path),
        "-m", message,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        st.error("❌ La commande `kaggle` n'est pas disponible sur ce serveur.")
        return False
    except subprocess.CalledProcessError as e:
        st.error("❌ Erreur lors de la soumission sur Kaggle.")
        st.code(e.stderr or e.stdout, language="bash")
        return False

    st.code(result.stdout, language="bash")
    st.success("📤 Submission envoyée sur Kaggle avec succès !")
    return True


def page_login(interactions_df):
    """Page de login avec ciel animé (étoiles + fusées)."""
    import random

    # Particules légères (halo bleu)
    particles_html = ""
    for i in range(10):
        top = random.randint(5, 95)
        left = random.randint(5, 95)
        delay = random.uniform(0, 4)
        duration = random.uniform(3, 6)
        particles_html += (
            f'<div class="particle" style="top:{top}%; left:{left}%; '
            f'animation-delay:{delay}s; animation-duration:{duration}s;"></div>'
        )

    # Étoiles fixes qui scintillent
    stars_html = ""
    for i in range(40):
        top = random.randint(5, 95)
        left = random.randint(0, 100)
        delay = random.uniform(0, 3)
        duration = random.uniform(2, 5)
        stars_html += (
            f'<div class="star" style="top:{top}%; left:{left}%; '
            f'animation-delay:{delay}s; animation-duration:{duration}s;"></div>'
        )

    # Injection dans la page
    st.markdown(particles_html + stars_html, unsafe_allow_html=True)

    # ============ GIF tout en haut, plus petit + flottant ============
    if GIF_PATH.exists():
        st.markdown(
            f"""
            <div class="floating-gif">
                <img src="data:image/gif;base64,{base64.b64encode(open(GIF_PATH, "rb").read()).decode()}"
                    width="160"/>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"GIF non trouvé : {GIF_PATH}")

    # ============ Carte titre AI LIBRARIAN ============
    st.markdown(
        """
        <div class="login-container">
            <div class="login-card" style="margin-top:0.3rem; margin-bottom:0.3rem;">
                <div class="login-title">🧠 AI LIBRARIAN</div>
                <div class="login-subtitle">Votre bibliothèque intelligente personnalisée</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============ Formulaire de login avec autocomplétion et création de nouveau user ============
    user_ids = sorted(interactions_df["u"].unique().tolist())
    
    # Initialiser le mode de login dans session_state
    if 'login_mode' not in st.session_state:
        st.session_state['login_mode'] = 'existing'  # 'existing' ou 'new'

    col1, col2, col3 = st.columns([1.5, 1.5, 1.5])

    with col2:
        # Toggle entre utilisateur existant et nouveau
        mode_tabs = st.tabs(["👤 Utilisateur existant", "✨ Nouveau utilisateur"])
        
        with mode_tabs[0]:
            # Mode utilisateur existant avec autocomplétion
            st.markdown("<p style='text-align:center; font-size:0.9rem; color:#9ca3af; margin-bottom:0.5rem;'>Sélectionnez votre User ID</p>", unsafe_allow_html=True)
            
            # Selectbox avec autocomplétion
            selected_user = st.selectbox(
                "User ID",
                options=[""] + user_ids,
                format_func=lambda x: "Choisissez un utilisateur..." if x == "" else f"User N° {x}",
                key="login_selectbox",
                label_visibility="collapsed"
            )
            
            login_button = st.button(
                "🚀 Connexion",
                type="primary",
                use_container_width=True,
                key="login_existing"
            )
            
            if login_button:
                if selected_user == "":
                    st.error("⚠️ Sélectionnez un User ID pour continuer")
                else:
                    st.session_state['logged_in_user'] = selected_user
                    st.session_state['page'] = 'home'
                    st.success(f"✅ Bienvenue User N° **{selected_user}** !")
                    st.balloons()
                    st.rerun()
        
        with mode_tabs[1]:
            # Mode création de nouveau user
            st.markdown("<p style='text-align:center; font-size:0.9rem; color:#9ca3af; margin-bottom:0.5rem;'>Créez un nouveau profil</p>", unsafe_allow_html=True)
            
            # Générer un nouvel ID (max + 1)
            new_user_id = max(user_ids) + 1 if user_ids else 1
            
            st.markdown(
                f"""
                <div style="text-align: center; padding: 1rem; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; margin-bottom: 1rem;">
                    <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.3rem;">Votre nouveau User ID sera :</div>
                    <div style="font-size: 1.8rem; font-weight: 700; background: linear-gradient(120deg, #60a5fa, #a855f7); -webkit-background-clip: text; color: transparent;">
                        {new_user_id}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            create_button = st.button(
                "✨ Créer mon profil",
                type="primary",
                use_container_width=True,
                key="create_new_user"
            )
            
            if create_button:
                interactions_path = INTERACTIONS_FILE  # défini en haut de ton app

                # Charger interactions_train.csv
                try:
                    inter_df = pd.read_csv(interactions_path)
                except FileNotFoundError:
                    inter_df = pd.DataFrame(columns=["u", "i", "r"])

                # Ajouter une interaction vide : i = -1 (signifie "aucun livre"), r = 0
                empty_interaction = pd.DataFrame([{
                    "u": new_user_id,
                    "i": 0,
                    "r": 0
                }])

                # Fusion + suppression des duplicates (au cas où)
                inter_df = pd.concat([inter_df, empty_interaction], ignore_index=True)
                inter_df = inter_df.drop_duplicates(subset=["u", "i"], keep="first")

                # Sauvegarde
                inter_df.to_csv(interactions_path, index=False)


                # Connexion automatique
                st.session_state['logged_in_user'] = new_user_id
                st.session_state['page'] = 'profil'
                st.session_state['is_new_user'] = True

                st.success(f"✨ Nouveau profil créé ! Bienvenue User N° {new_user_id}")
                st.balloons()
                st.rerun()

                # Petit texte d'info
                st.markdown(
                    f"""
                    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top:1rem;">
                        📊 <strong>{len(user_ids):,}</strong> utilisateurs dans la base
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# ==========================
#    SIDEBAR NAVIGATION
# ==========================
def render_sidebar(logged_in_user):
    """Affiche la sidebar avec navigation et info utilisateur"""
    
    with st.sidebar:
        # Info utilisateur
        st.markdown(
            f"""
            <div class="sidebar-user-info">
                <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem;">👤 Connecté en tant que</div>
                <div class="sidebar-user-id">User N° {logged_in_user}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Navigation
        
        
        current_page = st.session_state.get('page', 'home')
        
        if st.button("🏠 Accueil", key="nav_home", use_container_width=True):
            st.session_state['page'] = 'home'
            st.rerun()
        
        if st.button("📖 Voir mes recommandations", key="nav_reco", use_container_width=True):
            st.session_state['page'] = 'reco'
            st.rerun()
        
        if st.button("🧠 Générer nouvelles recommandations", key="nav_profil", use_container_width=True):
            st.session_state['page'] = 'profil'
            st.rerun()
        if st.button("🏆 Kaggle / Modèle global", key="nav_train", use_container_width=True):
            st.session_state['page'] = 'train'
            st.rerun()
        
        st.markdown("---")
        
        # Bouton de déconnexion
        if st.button("🚪 Déconnexion", key="logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        # Footer
        st.markdown(
            """
            <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(148,163,184,0.2); text-align: center; font-size: 0.75rem; color: #64748b;">
                <p>🤖 AI Librarian v2.0</p>
                <p>Powered by Zakaria, Yohan, & Pablo for Machine Learning course at HEC</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================
#        DATA LOADING
# ==========================
# @st.cache_data
def load_submission():
    try:
        df = pd.read_csv(SUBMISSION_FILE)
        return df
    except FileNotFoundError:
        st.error(f"❌ Fichier submission introuvable : {SUBMISSION_FILE}")
        return None


@st.cache_data
def load_items():
    try:
        df = pd.read_csv(ITEMS_FILE)
        return df
    except FileNotFoundError:
        st.error(f"❌ Fichier items introuvable : {ITEMS_FILE}")
        return None



def load_interactions():
    try:
        df = pd.read_csv(INTERACTIONS_FILE)
        return df
    except FileNotFoundError:
        st.error(f"❌ Fichier interactions introuvable : {INTERACTIONS_FILE}")
        return None


# ==========================
#  UTIL: PARSER & SERVICE
# ==========================
def parse_recommendation_string(rec_str):
    """Parse l'historique des recos -> retourne liste de listes (batches)."""
    if pd.isna(rec_str):
        return []

    # Exemple :
    # "2871 2873 7818 9645 7554, 9644 2875 2874 3845 9628"
    batches = [b.strip() for b in str(rec_str).split(",")]

    parsed_batches = []
    for batch in batches:
        ids = []
        for token in batch.replace(",", " ").split():
            token = token.strip()
            if token.isdigit():
                ids.append(int(token))
        if ids:
            parsed_batches.append(ids)

    return parsed_batches
def kaggle_list_submissions():
    """
    Récupère les submissions Kaggle pour la compétition et les retourne en DataFrame.
    Nécessite :
      - kaggle CLI installé
      - ~/.kaggle/kaggle.json configuré (manuellement, pas via l'app)
    """
    cmd = [
        "kaggle", "competitions", "submissions",
        "-c", KAGGLE_COMP,
        "--csv",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        st.error(
            "❌ La commande `kaggle` n'est pas disponible sur ce serveur.\n\n"
            "Configure-la sur la machine locale (ou sur Streamlit Cloud) avec `kaggle.json`."
        )
        return None
    except subprocess.CalledProcessError as e:
        st.error("❌ Erreur lors de la récupération des submissions Kaggle.")
        st.code(e.stderr or e.stdout, language="bash")
        return None

    if not result.stdout.strip():
        st.info("Aucune submission trouvée sur Kaggle.")
        return None

    try:
        df = pd.read_csv(io.StringIO(result.stdout))
        return df
    except Exception as e:
        st.error(f"Impossible de parser les submissions Kaggle : {e}")
        # On montre un extrait brut pour debug, mais PAS du HTML ni autre gadget.
        st.code(result.stdout[:1000], language="text")
        return None

def fake_reco_service(sub_df, items_df, user_id, liked_item_ids, top_k=10):
    """
    Mock de service de reco.
    """
    liked_item_ids = list(dict.fromkeys(liked_item_ids))  # unique, ordre conservé

    all_ids = items_df["i"].dropna().astype(int).unique()
    all_ids_sorted = sorted(all_ids)

    reco_ids = liked_item_ids.copy()
    for item_id in all_ids_sorted:
        if len(reco_ids) >= top_k:
            break
        if item_id not in reco_ids:
            reco_ids.append(item_id)

    reco_ids = reco_ids[:top_k]
    rec_str = " ".join(str(x) for x in reco_ids)

    if user_id in sub_df["user_id"].values:
        sub_df.loc[sub_df["user_id"] == user_id, "recommendation"] = rec_str
    else:
        sub_df = pd.concat(
            [sub_df, pd.DataFrame([{"user_id": user_id, "recommendation": rec_str}])],
            ignore_index=True,
        )

    sub_df.to_csv(SUBMISSION_FILE, index=False)
    return sub_df, reco_ids


def display_enriched_book_card(row, item_id, index=None):
    """
    Affiche une carte enrichie pour un livre avec toutes ses métadonnées.
    
    Args:
        row: Ligne du DataFrame items_df
        item_id: ID du livre
        index: Numéro d'ordre (optionnel, pour l'animation)
    """
    # Récupération brute des données
    title = row.get("Title", f"Item {item_id}")
    author = row.get("Author", "")
    publisher = row.get("Publisher", "")
    summary = row.get("summary", "")
    raw_pub_year = row.get("published_year", "")
    raw_page_count = row.get("page_count", "")
    language = row.get("language", "")
    subjects = row.get("Subjects", "")
    
    # --- Nettoyage des champs texte ---
    title = title if pd.notna(title) and str(title).strip() else f"(Sans titre) [id {item_id}]"
    author = author if pd.notna(author) and str(author).strip() else ""
    publisher = publisher if pd.notna(publisher) and str(publisher).strip() else ""
    summary = summary if pd.notna(summary) and str(summary).strip() else ""
    language = language if pd.notna(language) and str(language).strip() else ""
    subjects = subjects if pd.notna(subjects) and str(subjects).strip() else ""

    # --- Année de publication (gérer date complète, int, string, etc.) ---
    pub_year = ""
    if pd.notna(raw_pub_year):
        s = str(raw_pub_year).strip()
        if s and s.lower() != "nan":
            # Si c'est une date style "2013-03-15" -> on prend les 4 premiers caractères
            # Si c'est déjà "2013" -> ça marche aussi
            if len(s) >= 4 and s[:4].isdigit():
                pub_year = s[:4]

    # --- Nombre de pages (robuste) ---
    page_count = ""
    if pd.notna(raw_page_count):
        s = str(raw_page_count).strip()
        if s and s.lower() != "nan":
            try:
                val = float(s)
                if val > 0:
                    page_count = str(int(val))
            except ValueError:
                page_count = ""

    # --- Langue → drapeau ---
    lang_flags = {"fr": "🇫🇷", "en": "🇬🇧", "de": "🇩🇪", "it": "🇮🇹", "es": "🇪🇸"}
    lang_display = ""
    if language:
        code = language.lower()
        lang_display = f"{lang_flags.get(code, '🌐')} {code.upper()}"

    # --- Construction de la ligne de métadonnées ---
    meta_parts = []
    if author:
        meta_parts.append(f"✍️ {author}")
    if pub_year:
        meta_parts.append(f"📅 {pub_year}")
    if page_count:
        meta_parts.append(f"📄 {page_count} pages")
    if lang_display:
        meta_parts.append(lang_display)

    meta_str = " · ".join(meta_parts) if meta_parts else "Métadonnées non disponibles"

    # --- Éditeur ---
    publisher_html = ""
    if publisher:
        publisher_html = (
            '<div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">'
            f"🏢 {publisher}"
            "</div>"
        )

    # --- Résumé (tronqué) ---
    summary_html = ""
    if summary:
        summary_truncated = summary[:250] + "..." if len(summary) > 250 else summary
        summary_html = f'''
        <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.6; margin-top:12px; padding:10px; background:rgba(15,23,42,0.5); border-left:3px solid #3b82f6; border-radius:4px;">
            📖 {summary_truncated}
        </div>
        '''

    # --- Badges de sujets ---
    subjects_html = ""
    if subjects:
        subject_list = [s.strip() for s in str(subjects).split(";") if s.strip()][:4]
        if subject_list:
            badges = "".join([f'<span class="book-badge">{s}</span>' for s in subject_list])
            subjects_html = f'<div style="margin-top:10px; margin-bottom:8px;">{badges}</div>'

    # --- Animation delay & numéro d'ordre ---
    delay = f"animation-delay: {index * 0.08}s;" if index is not None else ""
    order_html = (
        f'<div style="position:absolute; top:12px; right:12px; font-size:1.3rem; opacity:0.25; font-weight:700;">#{index}</div>'
        if index is not None
        else ""
    )

    # --- Affichage final ---
    st.markdown(
        f"""
        <div class="book-card" style="{delay}">
            {order_html}
            <div class="book-title">{title}</div>
            <div class="book-meta">{meta_str}</div>
            {publisher_html}
            {subjects_html}
            <div style="margin-top:6px;">
                <span class="book-badge" style="background:rgba(59,130,246,0.3);">ID {item_id}</span>
            </div>
            {summary_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
#      PAGE 1 : RECO
# ==========================

def page_reco_par_user(sub_df, items_df, interactions_df, logged_in_user):
    st.markdown("## 📖 Mes recommandations personnalisées")
    
    user_id_to_use = logged_in_user

    # Vérifier que l'user existe (dans interactions)
    if user_id_to_use not in interactions_df["u"].unique():
        st.error(f"❌ Votre User ID ({user_id_to_use}) n'existe pas dans la base de données.")
        return

    # Récupérer la ligne dans submission pour cet user
    user_rows = sub_df[sub_df["user_id"] == user_id_to_use]

    if user_rows.empty:
        st.warning("😕 Aucune recommandation trouvée pour votre profil.")
        st.info("💡 Génère tes premières recommandations via **'🧠 Générer nouvelles recommandations'**")
        return

    rec_str = str(user_rows.iloc[0]["recommendation"]).strip()

    # ================================
    # 1️⃣ Parser UNE SEULE LISTE + dédoublonner
    # ================================
    # On tolère encore les vieilles virgules au cas où, mais on ne s'en sert plus.
    tokens = rec_str.replace(",", " ").split()

    seen = set()
    rec_ids = []
    for tok in tokens:
        tok = tok.strip()
        if tok.isdigit():
            val = int(tok)
            if val not in seen:
                seen.add(val)
                rec_ids.append(val)

    if not rec_ids:
        st.info("Aucune recommandation disponible.")
        return

    # ================================
    # 2️⃣ Récupération des livres dans items.csv
    # ================================
    books = (
        items_df[items_df["i"].isin(rec_ids)]
        .copy()
        .set_index("i")
        .reindex(rec_ids)  # garde l'ordre d'apparition
    )

    st.markdown("### ✨ Tes livres recommandés")
    st.markdown(
        f"<p style='color:#9ca3af; font-size:0.9rem;'>📚 Nombre de recommandations : {len(rec_ids)}</p>",
        unsafe_allow_html=True,
    )

    # ================================
    # 3️⃣ Affichage des cartes
    # ================================
    for idx, (item_id, row) in enumerate(books.iterrows(), 1):
        display_enriched_book_card(row, item_id, index=idx)
# ==========================
#   PAGE 0 : ACCUEIL
# ==========================
def page_home(sub_df, items_df, interactions_df, logged_in_user):
    col_left, col_right = st.columns([4, 1])

    with col_left:
        st.markdown(
            f"""
            <div class="welcome-message" style="text-align:left; padding-bottom:1rem;">
                <div class="welcome-title" style="font-size:2.3rem;">
                    Bienvenue User N° {logged_in_user}
                </div>
                <div class="welcome-subtitle">
                    🎉 Ravi de te revoir dans ta bibliothèque intelligente !
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        if GIF_PATH.exists():
            st.markdown(
            f"""
            <div class="floating-gif">
                <img src="data:image/gif;base64,{base64.b64encode(open(GIF_PATH, "rb").read()).decode()}"
                    width="160"/>
            </div>
            """,
            unsafe_allow_html=True
        )
        else:
            st.warning(f"GIF non trouvé : {GIF_PATH}")

    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-chip">
                <span>🧠 AI Librarian</span>
                <span style="opacity:0.65;">Kaggle Recommender System</span>
            </div>
            <div class="hero-subtitle">
                Laisse le modèle d'IA te suggérer des livres, ou entraîne-le avec tes propres coups de cœur.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div style="font-size:2rem; color:#9ca3af;">👥 Utilisateurs (interactions)</div>
                <div style="font-size:2rem; font-weight:700;">{interactions_df['u'].nunique()}</div>
                <div style="font-size:0.9rem; color:#6b7280; margin-top:0.35rem;">
                    profils avec historique
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        score_html = """
            <div class="stat-card" style="text-align:center; position:relative;">
                <div style="
                    position:absolute;
                    top:8px;
                    right:8px;
                    background:linear-gradient(135deg, #fbbf24, #f59e0b);
                    padding:2px 8px;
                    border-radius:8px;
                    font-size:1rem;
                    font-weight:700;
                    color:#1f2937;
                    box-shadow:0 0 10px rgba(251,191,36,0.35);
                    display:flex;
                    align-items:center;
                    gap:4px;
                ">
                    🏆 2e Kaggle
                </div>
                <br>
                <div style="font-size:2rem; color:#9ca3af;">💪 Performance du modèle</div>
                <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:0.5rem;">
                    <div style="font-size:0.8rem; color:#9ca3af;">Score du modèle</div>
                </div>
                <div class="score-ring">
                    <div class="score-ring-inner">
                        <div class="score-ring-value"
                            style="
                                font-size:1.6rem;
                                background:none !important;
                                -webkit-text-fill-color:#FBBF24 !important;
                                color:#FBBF24 !important;
                            ">
                            0.17689
                        </div>
                    </div>
                </div>
                <div style="margin-top:0.55rem; font-size:1rem; color:#64748b; line-height:1.4;">
                    <div><strong>Stage 1</strong> · SBERT · ALS · SASRec · Markov · LightGCN</div>
                    <div style="margin-top:2px;"><strong>Stage 2</strong> · CatBoost · </div>
                </div>
            </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div style="font-size:2rem; color:#9ca3af;">📚 Livres possibles</div>
                <div style="font-size:2rem; font-weight:700;">{items_df.shape[0]}</div>
                <div style="font-size:0.9rem; color:#6b7280; margin-top:0.35rem;">
                    éléments dans la base items
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='cta-row'></div>", unsafe_allow_html=True)

    user_interactions = interactions_df[
        (interactions_df['u'] == logged_in_user) &
        (interactions_df['i'] != 0)
    ]
    if not user_interactions.empty:
        st.markdown("---")
        st.markdown(f"### 📊 Ton activité")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📚 Livres empruntés", len(user_interactions))
        with col2:
            user_recs = sub_df[sub_df['user_id'] == logged_in_user]
            has_recs = not user_recs.empty
            st.metric("✨ Recommandations disponibles", "Oui" if has_recs else "Non")
    
    st.markdown("---")
    st.markdown("### 🚀 Actions rapides")
    st.markdown("Utilise la navigation dans la barre latérale pour explorer tes recommandations ou en générer de nouvelles !")


# ==========================
#   PAGE 2 : PROFIL & RECO
# ==========================
def page_construire_profil(sub_df, items_df, interactions_df, logged_in_user):
    st.markdown("## 🧠 Générer nouvelles recommandations")

    st.markdown(
        """
        **🎯 Comment ça marche :**
        1. Sélectionne des livres que tu aimes
        2. Le modèle IA génère tes recommandations personnalisées basées sur tes goûts !
        """
    )

    # ==========================
    # 0) Init session + user
    # ==========================
    if "selected_book_ids" not in st.session_state:
        st.session_state["selected_book_ids"] = []

    selected_user_id = logged_in_user
    st.markdown(f"### 1️⃣ Ton profil : User N° {selected_user_id}")

    # Historique utilisateur
    user_history = interactions_df[
        (interactions_df["u"] == selected_user_id) & (interactions_df["i"] > 0)
    ]
    if not user_history.empty:
        st.info(f"📚 Cet utilisateur a déjà **{len(user_history)} interactions** dans l'historique")
        with st.expander(f"Voir l'historique de l'utilisateur {selected_user_id}"):
            history_items = user_history["i"].tolist()
            history_books = items_df[items_df["i"].isin(history_items)][["i", "Title", "Author"]].head(10)
            render_pretty_table(history_books, max_height=260)

    st.markdown("---")
    st.markdown("### 2️⃣ Sélectionne tes livres préférés")
    st.markdown("**💡 Astuce :** Plus tu sélectionnes de livres, plus les recommandations seront précises !")

    # ==========================
    # 1) Recherche + tableau cliquable
    # ==========================
    search_term = st.text_input("🔍 Rechercher par titre, auteur ou sujet :", "")

    items_view = items_df.copy()
    items_view["Title"] = items_view["Title"].fillna("(Sans titre)")
    items_view["Author"] = items_view["Author"].fillna("")
    if "Subjects" not in items_view.columns:
        items_view["Subjects"] = ""

    if search_term.strip():
        s = search_term.lower()
        items_view = items_view[
            items_view["Title"].astype(str).str.lower().str.contains(s)
            | items_view["Author"].astype(str).str.lower().str.contains(s)
            | items_view["Subjects"].astype(str).str.lower().str.contains(s)
        ]

    max_rows = st.slider("Nombre de livres à afficher :", 10, 100, 30, step=10)

    cols_to_show = ["i", "Title", "Author", "published_year"]
    cols_to_show = [c for c in cols_to_show if c in items_view.columns]

    subset = items_view[cols_to_show].head(max_rows).copy()
    subset["i"] = subset["i"].astype(int)

    st.markdown("#### 📚 Résultats (cliquables)")

    # Tableau compact avec bouton "⭐ Ajouter" par ligne
    for _, row in subset.iterrows():
        book_id = int(row["i"])
        title = str(row.get("Title", f"ID {book_id}"))
        author = str(row.get("Author", "") or "")
        year = str(row.get("published_year", "") or "")

        col_info, col_btn = st.columns([6, 1])
        with col_info:
            line = f"**{title}**"
            if author.strip():
                line += f" — *{author}*"
            if year.strip():
                line += f" · ({year})"
            line += f" · ID `{book_id}`"
            st.markdown(line)

        with col_btn:
            if book_id in st.session_state["selected_book_ids"]:
                st.button("✅ Ajouté", key=f"added_{book_id}", disabled=True)
            else:
                if st.button("⭐ Ajouter", key=f"add_{book_id}"):
                    st.session_state["selected_book_ids"].append(book_id)
                    st.rerun()

    # ==========================
    # 2) Récap sélection
    # ==========================
    st.markdown("---")
    st.markdown("### 3️⃣ Ta sélection")

    selected_book_ids = st.session_state["selected_book_ids"]

    if not selected_book_ids:
        st.info("Clique sur un livre dans la liste pour l'ajouter à ta sélection.")
    else:
        st.success(f"✅ **{len(selected_book_ids)} livre(s)** sélectionné(s)")
        if st.button("🧹 Vider ma sélection"):
            st.session_state["selected_book_ids"] = []
            st.rerun()

        selected_df = items_df[
            items_df["i"].isin(selected_book_ids)
        ][["i", "Title", "Author", "published_year"]]
        render_pretty_table(selected_df, max_height=220)

    # ==========================
    # 3) Générer les reco IA
    # ==========================
    st.markdown("---")
    st.markdown("### 4️⃣ Générer les recommandations IA")

    if st.button(
        "🚀 Générer mes recommandations personnalisées",
        key="btn_run_ml_single",
        type="primary",
        disabled=not selected_user_id
    ):
        if selected_user_id is None:
            st.warning("⚠️ Utilisateur non connecté.")
            return
        if not selected_book_ids:
            st.warning("⚠️ Sélectionne au moins un livre pour personnaliser tes recommandations !")
            return

        # -------- 1) MAJ interactions_train.csv --------
        try:
            inter_df = pd.read_csv(INTERACTIONS_FILE)
        except FileNotFoundError:
            inter_df = pd.DataFrame(columns=["u", "i", "r"])

        # Supprimer placeholders i=0 pour cet user
        if not inter_df.empty:
            inter_df = inter_df[
                ~((inter_df["u"] == selected_user_id) & (inter_df["i"] == 0))
            ]

        new_interactions = pd.DataFrame(
            [{"u": selected_user_id, "i": int(bid), "r": 1} for bid in selected_book_ids]
        )

        if not inter_df.empty:
            inter_df = pd.concat([inter_df, new_interactions], ignore_index=True)
            inter_df = inter_df.drop_duplicates(subset=["u", "i"], keep="last")
        else:
            inter_df = new_interactions

        inter_df.to_csv(INTERACTIONS_FILE, index=False)

        single_user_script = ROOT_DIR / "generate_single_user_reco.py"
        if not single_user_script.exists():
            st.error(f"❌ Script introuvable : {single_user_script}")
            return

        python_cmd = sys.executable

        st.markdown(
            f"""
            <div class="loader-wrapper">
                <div class="loader-orb"></div>
                <p style="margin-top:0.75rem; color:#9ca3af; font-size:0.9rem;">
                    🧠 <strong>AI LIBRARIAN</strong> • Génération pour l'utilisateur <strong>{selected_user_id}</strong>...
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress_container = st.container()

        model_steps = {
            "Loading data": {"label": "📊 Chargement des données", "progress": 0},
            "Loading enriched book data": {"label": "📚 Chargement métadonnées", "progress": 0},
            "Loading embeddings": {"label": "🧬 Chargement embeddings", "progress": 0},
            "Loading CatBoost": {"label": "🤖 Chargement modèles ML", "progress": 0},
            "Getting top": {"label": "� Génération des candidats", "progress": 0},
            "Re-ranking": {"label": "🎲 Re-ranking intelligent", "progress": 0},
            "Top 10 recommandations": {"label": "✨ Finalisation top 10", "progress": 0},
            "Updating submission": {"label": "💾 Mise à jour submission.csv", "progress": 0},
        }

        progress_bars = {}
        status_texts = {}

        with progress_container:
            for step_key, step_info in model_steps.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    status_texts[step_key] = st.empty()
                    status_texts[step_key].markdown(
                        f"<span style='color:#64748b; font-size:0.85rem;'>⏸️ {step_info['label']}</span>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    progress_bars[step_key] = st.progress(0)

        with st.expander("🔧 Logs techniques (pour les hackers)", expanded=False):
            log_area = st.empty()

        # Commande du script
        command = [python_cmd, str(single_user_script), str(selected_user_id)]
        command.extend([str(bid) for bid in selected_book_ids])

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT_DIR),
            universal_newlines=True,
        )

        logs = []
        completed_steps = set()

        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                line = line.rstrip()
                logs.append(line)

                # Maj des steps
                for step_key, step_info in model_steps.items():
                    if step_key in line and step_key not in completed_steps:
                        status_texts[step_key].markdown(
                            f"<span style='color:#3b82f6; font-size:0.85rem;'>⏳ {step_info['label']} <strong>EN COURS...</strong></span>",
                            unsafe_allow_html=True,
                        )
                        progress_bars[step_key].progress(50)

                        for prev_key in list(model_steps.keys()):
                            if prev_key == step_key:
                                break
                            if prev_key not in completed_steps:
                                status_texts[prev_key].markdown(
                                    f"<span style='color:#22c55e; font-size:0.85rem;'>✅ {model_steps[prev_key]['label']}</span>",
                                    unsafe_allow_html=True,
                                )
                                progress_bars[prev_key].progress(100)
                                completed_steps.add(prev_key)
                        break

                log_area.code("\n".join(logs[-50:]), language="bash")

        # Fin du process
        for line in process.stdout:
            line = line.rstrip()
            logs.append(line)

        for step_key, step_info in model_steps.items():
            status_texts[step_key].markdown(
                f"<span style='color:#22c55e; font-size:0.85rem;'>✅ {step_info['label']}</span>",
                unsafe_allow_html=True,
            )
            progress_bars[step_key].progress(100)

        log_area.code("\n".join(logs[-50:]), language="bash")

        returncode = process.returncode

        if returncode != 0:
            st.error("❌ Erreur lors de l'exécution du modèle.")
            with st.expander("📋 Voir tous les logs d'erreur"):
                st.code("\n".join(logs), language="bash")
            return

        st.markdown(
            f"""
            <div style="text-align:center; padding:2rem 0;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🎉</div>
                <div style="font-size:1.5rem; font-weight:700; background: linear-gradient(120deg, #22c55e, #3b82f6); -webkit-background-clip: text; color: transparent;">
                    RECOMMANDATIONS GÉNÉRÉES
                </div>
                <div style="color:#9ca3af; margin-top:0.5rem;">Top 10 livres pour l'utilisateur <strong>{selected_user_id}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Rechargement du nouveau submission.csv
        new_sub_df = load_submission()

        if new_sub_df is not None:
            user_recs = new_sub_df[new_sub_df["user_id"] == selected_user_id]
            if not user_recs.empty:
                rec_str = user_recs.iloc[0]["recommendation"]
                rec_ids = [int(x) for x in str(rec_str).split() if str(x).strip().isdigit()]

                st.success(f"✅ **{len(rec_ids)} recommandations** générées pour l'utilisateur {selected_user_id}")

                if len(rec_ids) > 0:
                    st.markdown("### 📚 Top 10 Recommandations")
                    books = (
                        items_df[items_df["i"].isin(rec_ids)]
                        .copy()
                        .set_index("i")
                        .reindex(rec_ids)
                    )

                    for idx, (item_id, row) in enumerate(books.iterrows(), 1):
                        display_enriched_book_card(row, item_id, index=idx)

        st.info("✨ Direction : **📖 Mes recommandations** pour voir toutes les suggestions !")

def get_kaggle_credentials():
    """
    Essaie de récupérer les credentials Kaggle via :
      1) Variables d'environnement KAGGLE_USERNAME / KAGGLE_KEY
      2) ~/.kaggle/kaggle.json
      3) kaggle.json dans le dossier du projet (ROOT_DIR/kaggle.json)
    """
    # 1) Env vars (solution propre si tu veux configurer ça via secrets + env)
    user = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if user and key:
        return user, key

    # 2) ~/.kaggle/kaggle.json
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        ROOT_DIR / "kaggle.json",   # 3) fichier dans le projet
        BASE_DIR / "kaggle.json",
    ]

    for cfg_path in candidates:
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                return cfg.get("username"), cfg.get("key")
            except Exception:
                continue

    return None, None

    """
    Affiche un leaderboard stylé avec couleurs selon le rang
    et met en évidence la ligne correspondant à my_team_name.
    """
    if df_lb is None or df_lb.empty:
        st.info("Leaderboard indisponible pour l'instant.")
        return

    # Normalisation colonnes
    rank_col = "Rank" if "Rank" in df_lb.columns else None
    team_col = "TeamName" if "TeamName" in df_lb.columns else "teamName"
    score_col = "Score" if "Score" in df_lb.columns else "score"

    df = df_lb.copy()

    # Si pas de Rank fourni, on le calcule
    if rank_col is None:
        if score_col in df.columns:
            df = df.sort_values(score_col, ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))
        rank_col = "Rank"

    # On limite à top 100 pour l'affichage
    df = df.head(100)

    rows_html = []
    my_rank_text = None

    for _, row in df.iterrows():
        rank = int(row.get(rank_col, 0))
        team_name = str(row.get(team_col, ""))
        score = row.get(score_col, "")

        # style de base
        bg = "rgba(15,23,42,0.8)"
        border = "rgba(55,65,81,1)"
        emoji = "🏁"

        # Couleurs selon rang
        if rank == 1:
            emoji = "🥇"
            bg = "linear-gradient(90deg,#facc15,#f97316)"
        elif rank == 2:
            emoji = "🥈"
            bg = "linear-gradient(90deg,#e5e7eb,#9ca3af)"
        elif rank == 3:
            emoji = "🥉"
            bg = "linear-gradient(90deg,#fed7aa,#fb923c)"
        elif rank <= 10:
            emoji = "🏆"
            bg = "rgba(22,163,74,0.25)"
        elif rank <= 50:
            emoji = "💡"
            bg = "rgba(37,99,235,0.12)"

        # Highlight ton équipe
        highlight = ""
        badge_html = ""
        if my_team_name and team_name == my_team_name:
            highlight = "box-shadow:0 0 18px rgba(59,130,246,0.7); border-color:#60a5fa;"
            badge_html = '<span style="margin-left:6px; padding:2px 8px; border-radius:999px; font-size:0.7rem; background:rgba(59,130,246,0.2); color:#bfdbfe;">⭐ Toi</span>'
            my_rank_text = f"{rank}"

        score_str = ""
        if pd.notna(score):
            try:
                score_str = f"{float(score):.5f}"
            except Exception:
                score_str = str(score)

        rows_html.append(f"""
            <tr style="background:{bg}; border-bottom:1px solid {border}; {highlight}">
                <td style="padding:8px 10px; font-weight:700; text-align:center;">{emoji}</td>
                <td style="padding:8px 10px; font-weight:700; text-align:center;">{rank}</td>
                <td style="padding:8px 10px; font-size:0.88rem;">{team_name}{badge_html}</td>
                <td style="padding:8px 10px; text-align:center; font-family:monospace;">{score_str}</td>
            </tr>
        """)

    st.markdown("### 🏆 Leaderboard de la compétition")

    if my_rank_text:
        st.markdown(
            f"""
            <div style="margin-bottom:0.6rem; font-size:0.9rem; color:#e5e7eb;">
                📊 Ton rang actuel : <strong>#{my_rank_text}</strong> ({my_team_name})
            </div>
            """,
            unsafe_allow_html=True
        )

    html = f"""
    <div class="nice-table-container" style="max-height:380px;">
        <table class="nice-table">
            <thead>
                <tr>
                    <th style="width:40px;">🏁</th>
                    <th>Rang</th>
                    <th>Équipe</th>
                    <th>Score public</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


    # On met des emojis + renomme les colonnes
    def rank_medal(r):
        try:
            r_int = int(r)
        except Exception:
            return str(r)
        if r_int == 1:
            return "🥇 1"
        elif r_int == 2:
            return "🥈 2"
        elif r_int == 3:
            return "🥉 3"
        elif r_int <= 10:
            return f"🏅 {r_int}"
        else:
            return f"{r_int}"

    if "rank" in df_lb.columns:
        df_lb["rank"] = df_lb["rank"].apply(rank_medal)

    if "score" in df_lb.columns:
        df_lb["score"] = df_lb["score"].apply(lambda s: f"{s:.5f}" if pd.notna(s) else "")

    rename_lb = {
        "rank": "🏆 Rang",
        "teamName": "👥 Équipe",
        "score": "🏅 Score",
        "entries": "📤 Submissions",
        "lastSubmission": "🕒 Dernier envoi",
    }
    df_lb = df_lb.rename(columns=rename_lb)

    st.markdown(
        "<p style='color:#9ca3af; font-size:0.9rem;'>Aperçu du classement global de la compétition Kaggle.</p>",
        unsafe_allow_html=True
    )
    # On ne montre que le top 30 pour que ça reste lisible en démo
    render_pretty_table(df_lb.head(30), max_height=380)

def render_kaggle_submissions_minimal(df_sub):
    if df_sub is None or df_sub.empty:
        st.info("Aucune submission trouvée.")
        return

    # On ne garde que la date
    cols = []
    if "date" in df_sub.columns:
        cols.append("date")
    if "description" in df_sub.columns:
        cols.append("description")
    if not cols:
        st.info("Impossible d'afficher les submissions (pas de colonne date).")
        return

    df = df_sub[cols].copy()

    # Séparer date et heure pour un affichage propre
    try:
        df["date"] = pd.to_datetime(df["date"])
        df["Jour"] = df["date"].dt.strftime("%Y-%m-%d")
        df["Heure"] = df["date"].dt.strftime("%H:%M:%S")
        df = df[["Jour", "Heure", "description"]] if "description" in df.columns else df[["Jour", "Heure"]]
    except:
        pass

    st.markdown("### 🕒 Historique des submissions")
    st.dataframe(df, height=300)
def page_train_full_model():
    st.markdown("## 🏆 Espace Kaggle : générer & envoyer le submission")

    st.markdown(
        """
        Ici tu peux :
        - 🚀 **générer un nouveau `submission.csv`** en lançant `create_submission.py`
        - 📤 **envoyer le `submission.csv` actuel sur Kaggle**
        - 📊 **voir simplement l’historique de tes submissions Kaggle**
        """
    )



    st.markdown("---")

    # ========== 2) Envoyer le fichier actuel sur Kaggle ==========
    st.markdown("### 2️⃣ Envoyer le `submission.csv` actuel sur Kaggle")

    if not SUBMISSION_FILE.exists():
        st.warning(
            "⚠️ Aucun fichier `submission.csv` trouvé dans le repo. "
            "Génère-le d'abord avec le bouton ci-dessus ou commite un fichier existant."
        )
    else:
        default_msg = "Submission from AI Librarian Streamlit app"
        submission_msg = st.text_input(
            "Message de submission (Kaggle)",
            value=default_msg,
            help="Ce message apparaîtra dans l'historique des submissions Kaggle."
        )

        if st.button("📤 Envoyer `submission.csv` sur Kaggle", key="btn_submit_kaggle_simple"):
            kaggle_submit_submission(SUBMISSION_FILE, submission_msg)

    st.markdown("---")

    # ========== 3) Historique des submissions ==========
    st.markdown("### 3️⃣ Historique de tes submissions Kaggle")

    st.markdown(
        "<p style='color:#9ca3af; font-size:0.9rem;'>"
        "Clique sur le bouton ci-dessous pour recharger l'historique à chaque fois que tu fais une nouvelle submission."
        "</p>",
        unsafe_allow_html=True,
    )

    # Le simple fait de cliquer relance le script Streamlit → ça suffit comme "refresh"
    st.button("🔄 Recharger les submissions Kaggle", key="btn_refresh_submissions")

    df_sub = kaggle_list_submissions()
    if df_sub is not None and not df_sub.empty:
        # Affichage simple mais propre (utilise ta fonction déjà stylée)
        render_pretty_table(df_sub, max_height=380)
    else:
        st.info("Aucune submission trouvée sur Kaggle (ou bien la commande Kaggle a échoué).")
# ==========================
#           MAIN
# ==========================
def main():
    sub_df = load_submission()
    items_df = load_items()
    interactions_df = load_interactions()

    if sub_df is None or items_df is None or interactions_df is None:
        st.stop()

    if 'recommendation' in sub_df.columns:
        rec_col = 'recommendation'
    elif 'item_id' in sub_df.columns:
        rec_col = 'item_id'
    else:
        st.error("Le CSV submission doit contenir 'user_id' et soit 'recommendation' soit 'item_id'")
        st.write("Colonnes trouvées dans submission :", list(sub_df.columns))
        st.stop()
    
    if rec_col == 'item_id':
        sub_df = sub_df.rename(columns={'item_id': 'recommendation'})

    if "i" not in items_df.columns:
        st.error("Le CSV items doit contenir la colonne 'i' (identifiant de livre).")
        st.write("Colonnes trouvées dans items :", list(items_df.columns))
        st.stop()
    
    if "u" not in interactions_df.columns:
        st.error("Le CSV interactions doit contenir la colonne 'u' (user_id).")
        st.write("Colonnes trouvées dans interactions :", list(interactions_df.columns))
        st.stop()

    if "logged_in_user" not in st.session_state:
        st.session_state["logged_in_user"] = None
    
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    logged_in_user = st.session_state.get("logged_in_user")
    
    if logged_in_user is None:
        page_login(interactions_df)
    else:
        render_sidebar(logged_in_user)
        
        current_page = st.session_state["page"]

        if current_page == "home":
            page_home(sub_df, items_df, interactions_df, logged_in_user)
        elif current_page == "reco":
            page_reco_par_user(sub_df, items_df, interactions_df, logged_in_user)
        elif current_page == "profil":
            page_construire_profil(sub_df, items_df, interactions_df, logged_in_user)
        elif current_page == "train":
            page_train_full_model()


if __name__ == "__main__":
    main()