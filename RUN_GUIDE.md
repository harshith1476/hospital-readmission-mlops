# Hospital Readmission MLOps: step-by-step commands

Dataset: `data/hospital_data_analysis.csv` (984 patients). Target: `Readmission` (Yes/No).
Features: Age, Gender, Condition, Cost, Length_of_Stay. Model: RandomForest. Quality gate: accuracy >= 0.90.
Shell: Windows PowerShell. Replace `<you>` with your GitHub username (`harshith1476`).

---

## 0. Setup (once)

```powershell
cd $HOME\Desktop
Expand-Archive hospital-readmission-mlops.zip -DestinationPath mlops-practicals
cd mlops-practicals\hospital-readmission-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
git --version ; docker --version
```

## 1. Quick local run of the whole project

```powershell
python train_model.py          # expect: Accuracy ~0.9898, creates readmission_model.pkl, metrics.json, hospital_clean.csv
python quality_gate.py         # expect: QUALITY GATE PASSED
python -m unittest discover -v # expect: Ran 13 tests ... OK
python app.py                  # API on http://localhost:5000  (leave running)
```

New PowerShell window, test the API:

```powershell
Invoke-RestMethod http://localhost:5000/health

$body = @{ Age=75; Gender="Female"; Condition="Heart Attack"; Cost=18000; Length_of_Stay=41 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
# expect prediction = Yes

$body = @{ Age=70; Gender="Male"; Condition="Hypertension"; Cost=1000; Length_of_Stay=35 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
# expect prediction = No
```

Stop the API with Ctrl+C.

---

## Practical 1: First CI pipeline (separate repo `first-ci-demo`)

Folder `first-ci-demo/` (delivered next to this project). Create an empty repo `first-ci-demo` on GitHub first.

```powershell
cd ..\first-ci-demo
python -m unittest discover -v                 # expect 3 tests OK
git init -b main
git add .
git commit -m "Add readmission logic, tests and first CI workflow"
git remote add origin https://github.com/<you>/first-ci-demo.git
git push -u origin main                        # Actions tab -> First Python CI -> green
```

Controlled failure, then fix:

```powershell
(Get-Content readmission_logic.py) -replace 'return "READMIT"','return "NO READMIT"' | Set-Content readmission_logic.py
git commit -am "Modify readmission logic"; git push        # red: test_high_risk_condition
git revert HEAD --no-edit; git push                                     # green again
```

---

## Practicals 2 to 7 (one repo, built up stage by stage)

Create an empty GitHub repo `hospital-readmission-mlops` (a new repo keeps your student-result repo `Mlops---CI-CD` untouched).
Each stage uses `stages\ml-ci.pN.yml` as that practical's workflow.

### Practical 2: ML training CI

```powershell
cd ..\hospital-readmission-mlops
git init -b main
git remote add origin https://github.com/<you>/hospital-readmission-mlops.git
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item stages\ml-ci.p2.yml .github\workflows\ml-ci.yml
git add .gitignore requirements.txt data train_model.py test_ml_pipeline.py .github
git commit -m "P2: hospital readmission ML training CI"
git push -u origin main
```

Check: Actions -> Hospital Readmission ML CI -> green, log shows `Number of records: 984`, 787 train / 197 test, 7 tests OK.

Failure demo:

```powershell
(Get-Content test_ml_pipeline.py) -replace 'predict\(patient\)\[0\]\), 1\)','predict(patient)[0]), 0)' | Set-Content test_ml_pipeline.py
git commit -am "Introduce test failure for CI demonstration"; git push     # red at Run automated ML tests
git revert HEAD --no-edit; git push                                       # green again
```

### Practical 3: Quality gate

```powershell
Copy-Item stages\ml-ci.p3.yml .github\workflows\ml-ci.yml
git add quality_gate.py .github
git commit -m "P3: add accuracy quality gate (min 0.90)"
git push
```

Failure demo (accuracy is 0.9898, so 0.999 fails):

```powershell
(Get-Content quality_gate.py) -replace 'MINIMUM_ACCURACY = 0.90','MINIMUM_ACCURACY = 0.999' | Set-Content quality_gate.py
git commit -am "Increase threshold for quality gate failure demo"; git push   # red at Check ML quality gate, exit code 1
git revert HEAD --no-edit; git push                                           # green
```

### Practical 4: Preserve ML artifacts

```powershell
Copy-Item stages\ml-ci.p4.yml .github\workflows\ml-ci.yml
git add .github
git commit -m "P4: preserve validated ML artifacts"
git push
```

Check: run summary -> Artifacts -> `hospital-readmission-ml-artifacts` (model, metrics.json, hospital_clean.csv).

Failure demo:

```powershell
(Get-Content .github\workflows\ml-ci.yml) -replace '^(\s+)readmission_model\.pkl','$1readmission_model_missing.pkl' | Set-Content .github\workflows\ml-ci.yml
git commit -am "Demonstrate missing artifact failure"; git push    # red at Upload validated ML artifacts
git revert HEAD --no-edit; git push
```

### Practical 5: Continuous delivery (Flask API + release candidate)

```powershell
Copy-Item stages\ml-ci.p5.yml .github\workflows\ml-ci.yml
git add app.py test_app.py .github
git commit -m "P5: Flask prediction API and release packaging"
git push
```

Check: 6 API tests OK, artifact `hospital-readmission-ml-release` (app.py, model, metrics.json, requirements.txt, DEPLOYMENT.txt).

Failure demo:

```powershell
(Get-Content test_app.py) -replace 'get_json\(\)\["prediction"\], "Yes"\)','get_json()["prediction"], "No")' | Set-Content test_app.py
git commit -am "Demonstrate application test failure"; git push    # red at Run prediction application tests
git revert HEAD --no-edit; git push
```

### Practical 6: Docker (manual build and run)

```powershell
Copy-Item stages\ml-ci.p6.yml .github\workflows\ml-ci.yml
git add Dockerfile .dockerignore .github
git commit -m "P6: Dockerfile and release candidate with Docker files"
git push
```

Then run Docker locally. Docker Desktop must be open and running.

```powershell
docker --version
docker info
docker run --rm hello-world

# the model file must exist in this folder (created by train_model.py in step 1)
python train_model.py
docker build -t hospital-readmission-ml:1.0 .
docker images
docker run -d --name hospital-readmission-container -p 5000:5000 hospital-readmission-ml:1.0
docker ps
```

Test the container:

```powershell
Invoke-RestMethod http://localhost:5000/health
$body = @{ Age=75; Gender="Female"; Condition="Heart Attack"; Cost=18000; Length_of_Stay=41 } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body $body
docker logs hospital-readmission-container
```

Optional port-mapping demo, then clean up:

```powershell
docker stop hospital-readmission-container; docker rm hospital-readmission-container
docker run -d --name no-port hospital-readmission-ml:1.0      # http://localhost:5000 will NOT open
docker stop no-port; docker rm no-port
docker run -d --name hospital-readmission-container -p 5000:5000 hospital-readmission-ml:1.0
docker stop hospital-readmission-container; docker rm hospital-readmission-container
```

If port 5000 is busy, use `-p 5001:5000` and `localhost:5001`.

### Practical 7: Docker build and test in GitHub Actions

```powershell
Copy-Item stages\ml-ci.p7.yml .github\workflows\ml-ci.yml
git add .github
git commit -m "P7: automate Docker build and container API tests"
git push
```

Check: all 16 steps green, `Wait for application to become ready` prints the health JSON, `Test prediction endpoint` prints `"prediction":"Yes"`.

Failure demo (wrong health-check port):

```powershell
(Get-Content .github\workflows\ml-ci.yml) -replace 'localhost:5000/health','localhost:5001/health' | Set-Content .github\workflows\ml-ci.yml
git commit -am "Demonstrate Docker health check failure"; git push   # red at Wait for application to become ready
git revert HEAD --no-edit; git push                                 # green
```

---

## Screenshots to take for your records

P1-P7 each need: green run, controlled red run with the expanded failing step, final green run. Plus P4 artifact list, P5 API test log, P6 `docker build`/`docker images`/`docker ps` and PowerShell predictions, P7 build and health/prediction logs.

## Notes

- Heart Attack and Heart Disease patients are readmitted about 100% of the time in this dataset, and each condition has a fixed Cost and Procedure. That is why accuracy is ~99%; treat the data as synthetic.
- `Procedure` is dropped (one-to-one with `Condition`). `Outcome` and `Satisfaction` are dropped because they are recorded after treatment.
- Condition and Gender are linked in the data (every Heart Attack patient is Female), so use real combinations when you test the API.
- Troubleshooting: `git push` rejected -> `git pull --rebase origin main` then push. `ModuleNotFoundError` locally -> activate `.venv`. Workflow YAML error -> check indentation in `.github\workflows\ml-ci.yml`.
