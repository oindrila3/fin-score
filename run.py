# ============================================================
# run.py — FIN-Score Pipeline Runner
# Windows-compatible alternative to Makefile
# ============================================================

import subprocess
import sys
import os

def run(command: str) -> None:
    """Runs a shell command and exits if it fails."""
    print(f"\n▶ Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {command}")
        sys.exit(1)

def install():
    """Install all dependencies."""
    print("📦 Installing dependencies...")
    run("pip install --only-binary :all: -r requirements.txt")
    print("✅ Dependencies installed")

def features():
    """Run feature engineering pipeline."""
    print("⚙️  Running feature engineering...")
    run("python -m src.features")
    print("✅ Features engineered")

def train():
    """Train both model variants."""
    print("🧠 Training models...")
    run("python -m src.train")
    print("✅ Models trained")

def predict():
    """Test scoring pipeline."""
    print("🎯 Testing scoring pipeline...")
    run("python -m src.predict")
    print("✅ Scoring pipeline working")

def monitor():
    """Run monitoring system."""
    print("📊 Running monitoring...")
    run("python -m src.monitoring")
    print("✅ Monitoring complete")

def pipeline():
    """Run full ML pipeline end to end."""
    print("\n🚀 Running full pipeline...")
    features()
    train()
    predict()
    monitor()
    print("\n✅ Full pipeline complete!")
    print("   Features → Models → Scoring → Monitoring")

def api():
    """Start FastAPI scoring endpoint."""
    print("🌐 Starting FIN-Score API...")
    print("📖 API docs: http://localhost:8000/docs")
    run("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

def dashboard():
    """Start Streamlit dashboard."""
    print("📊 Starting FIN-Score Dashboard...")
    run("streamlit run dashboard/app.py")

def test():
    """Run all tests."""
    print("🧪 Running all tests...")
    run("pytest tests/ -v")
    print("✅ All tests passed")

def clean():
    """Remove generated files."""
    print("🧹 Cleaning up...")
    run('find . -type f -name "*.pyc" -delete')
    run('find . -type d -name "__pycache__" -exec rm -rf {} +')
    print("✅ Cleanup complete")

def help():
    """Show available commands."""
    print("""
FIN-Score — B2B Lead Scoring System
=====================================

Usage: python run.py [command]

Setup:
  install       Install all dependencies

Pipeline:
  features      Run feature engineering
  train         Train both model variants
  predict       Test scoring pipeline
  monitor       Run monitoring system
  pipeline      Run full pipeline end to end

Serving:
  api           Start FastAPI endpoint
  dashboard     Start Streamlit dashboard

Testing:
  test          Run all tests

Cleanup:
  clean         Remove generated files
  help          Show this help message
""")

# ── Command Router ────────────────────────────────────────────
COMMANDS = {
    'install': install,
    'features': features,
    'train': train,
    'predict': predict,
    'monitor': monitor,
    'pipeline': pipeline,
    'api': api,
    'dashboard': dashboard,
    'test': test,
    'clean': clean,
    'help': help,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command not in COMMANDS:
        print(f"❌ Unknown command: {command}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[command]()

