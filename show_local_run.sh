#!/bin/zsh
cd "$(dirname "$0")" || exit 1
line() { print -- "\n========== $1 =========="; }
line "1. THIS MACHINE"
print "Computer : $(scutil --get ComputerName)"
print "Chip     : $(sysctl -n machdep.cpu.brand_string)"
print "RAM      : $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
sw_vers | head -2
line "2. PROJECT FOLDER"
pwd
line "3. PYTHON + PYTORCH (local venv, Apple GPU = MPS)"
.venv/bin/python -c "import sys, torch; print('Python', sys.version.split()[0]); print('PyTorch', torch.__version__); print('MPS (Apple GPU) available:', torch.backends.mps.is_available())"
line "4. TRAINING LOGS (device=mps means trained on this Mac's GPU)"
grep -h "device=" artifacts/logs/train_*.log
grep -h "done in" artifacts/logs/train_*.log
line "5. TRAINED MODEL FILES ON THIS DISK (date created)"
ls -lhT artifacts/runs/*/best.pt artifacts/best_model/model.pt
line "6. TEST RESULT OF SERVED MODEL"
.venv/bin/python -c "import json; c=json.load(open('artifacts/best_model/model_card.json')); t=c['test']['all']; print(f\"{c['model']}: test accuracy {t['accuracy']:.2%}, {t['errors']} errors in {t['n']} photos\")"
