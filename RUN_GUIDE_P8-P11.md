# Practicals 8, 9, 10, 11: Hospital Readmission API (one project)

Image: `ghcr.io/<you>/hospital-readmission-ml` · Port: 5000 · Health: `/health` · Shell: Windows PowerShell
Replace `<you>` with your GitHub username (lowercase, e.g. `harshith1476`). Docker Desktop must be running.
NEVER paste your token into files, screenshots or chat. Use it only inside your own terminal.

Project layout: `app.py` (P8-P10 version, no logging), `stages\` (P9 and P11 workflows + P11 logging app and tests), `test_payload.json`, `Dockerfile`.

---

## 0. Setup (once)

```powershell
cd $HOME\Desktop
Expand-Archive hospital-mlops-p8-11.zip -DestinationPath mlops-p8-11
cd mlops-p8-11\hospital-mlops-p8-11      # if `dir` does not show app.py, go one folder deeper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py                    # Accuracy ~0.9898, creates readmission_model.pkl
python quality_gate.py                   # QUALITY GATE PASSED
python -m unittest discover -v           # Ran 13 tests ... OK
```

Create an EMPTY GitHub repo named `hospital-readmission-ml` (no README). Then:

```powershell
git init -b main
git remote add origin https://github.com/<you>/hospital-readmission-ml.git
```

---

## Practical 8: Publish the Docker image to GHCR manually

### 8.1 Build and test locally
```powershell
docker build -t hospital-readmission-ml:v1 .
docker run -d --name hospital-readmission-container -p 5000:5000 hospital-readmission-ml:v1
Start-Sleep 3
Invoke-RestMethod http://localhost:5000/health
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -InFile test_payload.json
docker stop hospital-readmission-container; docker rm hospital-readmission-container
```
Expect `prediction = Yes`.

### 8.2 Create a token (in the browser)
GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic) -> Generate new token (classic).
Name `ghcr-hospital`, tick `write:packages` (read:packages is included), generate, copy it once.

### 8.3 Login, tag, push
```powershell
$env:GHCR_TOKEN = Read-Host "Paste token"
$env:GHCR_TOKEN | docker login ghcr.io -u <you> --password-stdin
docker tag hospital-readmission-ml:v1 ghcr.io/<you>/hospital-readmission-ml:v1
docker tag hospital-readmission-ml:v1 ghcr.io/<you>/hospital-readmission-ml:latest
docker push ghcr.io/<you>/hospital-readmission-ml:v1
docker push ghcr.io/<you>/hospital-readmission-ml:latest
Remove-Item Env:GHCR_TOKEN
```
Check: GitHub profile -> Packages -> `hospital-readmission-ml` shows tags `v1` and `latest`.

### 8.4 Pull and run from GHCR
```powershell
docker rmi ghcr.io/<you>/hospital-readmission-ml:latest
docker pull ghcr.io/<you>/hospital-readmission-ml:latest
docker run -d --name ghcr-test -p 5000:5000 ghcr.io/<you>/hospital-readmission-ml:latest
Start-Sleep 3
Invoke-RestMethod http://localhost:5000/health
docker stop ghcr-test; docker rm ghcr-test
```
Make the package public (needed for Render without credentials): Package page -> Package settings -> Change visibility -> Public.

Screenshots: docker build, docker push output, GHCR package page with tags, pull + health response.

---

## Practical 9: GitHub Actions builds, tests and publishes to GHCR

The workflow trains the model in CI (the model file is not committed), runs the gate and tests, builds the image, health-tests the container, then pushes `latest` and `sha-<commit>`.

```powershell
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item stages\docker-publish.p9.yml .github\workflows\docker-publish.yml
git add .
git status                                  # readmission_model.pkl / metrics.json must NOT be listed
git commit -m "P9: build, test and publish hospital image to GHCR"
git push -u origin main
```
Check: Actions -> Build, Test and Publish Hospital Readmission ML -> all 14 steps green. Packages now show `latest` and `sha-<commit>`.
If the push step is denied: repo Settings -> Actions -> General -> Workflow permissions -> Read and write; and on the package page -> Package settings -> Manage Actions access -> add the repo with Write role.

Failure demo (nothing is published when the container health test fails):
```powershell
(Get-Content .github\workflows\docker-publish.yml) -replace 'localhost:5000/health','localhost:5001/health' | Set-Content .github\workflows\docker-publish.yml
git commit -am "Demonstrate failed health check blocks publishing"; git push     # red at Wait for application health check; Log in/Push skipped
git revert HEAD --no-edit; git push                                              # green again
```

---

## Practical 10: Deploy the GHCR image to Render

Prerequisite: P9 run is green and the package is Public (8.4).

In the Render dashboard (browser):
1. New -> Web Service -> **Existing Image**.
2. Image URL: `ghcr.io/<you>/hospital-readmission-ml:latest` (private package: Add credential -> GitHub -> username + PAT classic with `read:packages` only).
3. Name: `hospital-readmission-ml`. Pick a Free/available instance.
4. Environment variable: `PORT` = `5000`.
5. Advanced -> Health Check Path: `/health`.
6. Deploy. Wait for Live and copy the URL `https://<service>.onrender.com`.

Test from PowerShell (the first request after sleep can take about 50 seconds on the free plan):
```powershell
$url = "https://<service>.onrender.com"
Invoke-RestMethod "$url/"
Invoke-RestMethod "$url/health"
Invoke-RestMethod -Uri "$url/predict" -Method Post -ContentType "application/json" -InFile test_payload.json
curl.exe --fail --silent "$url/health"
curl.exe -X POST "$url/predict" -H "Content-Type: application/json" --data "@test_payload.json"
```
This is a Flask app, so there is no `/docs` Swagger page; the endpoints are `/`, `/health`, `/predict`.

Logs: Render service -> Logs (startup + requests) and Events (image pull + deploy).

Update cycle (image services do NOT auto-redeploy):
```powershell
(Get-Content app.py) -replace 'APP_VERSION = "1.0"','APP_VERSION = "1.0-render-update"' | Set-Content app.py
git commit -am "Change API version for redeploy demo"; git push
```
Wait for the green Actions run, then Render -> Manual Deploy -> **Deploy latest reference**. Re-run `Invoke-RestMethod "$url/health"`: `version` is now `1.0-render-update`.

Screenshots: Actions run, GHCR package, Render config (no token), Live status, root, /health, /predict, Render logs, Manual Deploy.

---

## Practical 11: Logging and error tracking

Adds: stdout logging (INFO/WARNING/ERROR), model_loaded event, request ID + duration on every request, validation warnings, 404 warnings, prediction_failed with stack trace, no patient values in logs (only feature names; `SAFE_INPUT_FIELDS` is empty).

### 11.1 Apply the change on a branch
```powershell
git switch -c practical-11-logging
Copy-Item stages\app.p11.py app.py -Force
Copy-Item stages\test_logging.p11.py test_logging.py
Copy-Item stages\docker-publish.p11.yml .github\workflows\docker-publish.yml -Force
python -m unittest discover -v              # Ran 20 tests ... OK
```

### 11.2 Run locally and read the logs
```powershell
python app.py
```
Terminal shows `event=model_loaded model_file=readmission_model.pkl`. In a second PowerShell:
```powershell
Invoke-RestMethod http://localhost:5000/health                      # INFO http_request status=200
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -InFile test_payload.json
#   INFO event=prediction_request feature_names=[...] safe_input={}
#   INFO event=prediction_success prediction=Yes
#   INFO event=http_request ... status=200 duration_ms=...
try { Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{"Age":"abc"}' } catch { "400 as expected" }
#   WARNING event=validation_error ... ; WARNING http_request status=400
try { Invoke-RestMethod http://localhost:5000/not-a-real-route } catch { "404 as expected" }
#   WARNING http_request status=404
curl.exe -i http://localhost:5000/health                            # header X-Request-ID is present
```
Stop with Ctrl+C. (This Flask API returns 400 for bad input; the 422 in the FastAPI practical doc maps to 400 here.)

### 11.3 Docker test
```powershell
docker build -t hospital-readmission-ml:p11 .
docker run -d --name p11 -p 5000:5000 hospital-readmission-ml:p11
Start-Sleep 3
Invoke-RestMethod http://localhost:5000/health
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -InFile test_payload.json
try { Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType "application/json" -Body '{"Age":"abc"}' } catch { "400 as expected" }
try { Invoke-RestMethod http://localhost:5000/not-a-real-route } catch { "404 as expected" }
docker logs p11
docker stop p11; docker rm p11
```

### 11.4 Commit, push, merge (CI publishes the new image)
```powershell
git add app.py test_logging.py .github
git commit -m "Add API logging and error tracking"
git push -u origin practical-11-logging
git switch main
git merge practical-11-logging
git push
```
Check Actions: now 15 steps including "Test validation error and missing route"; package `latest` updated.

### 11.5 Redeploy on Render and read the logs
Render -> Manual Deploy -> **Deploy latest reference** -> wait for Live. Generate traffic:
```powershell
$url = "https://<service>.onrender.com"
Invoke-RestMethod "$url/health"
Invoke-RestMethod -Uri "$url/predict" -Method Post -ContentType "application/json" -InFile test_payload.json
try { Invoke-RestMethod -Uri "$url/predict" -Method Post -ContentType "application/json" -Body '{"Age":"abc"}' } catch { "400" }
try { Invoke-RestMethod "$url/not-a-real-route" } catch { "404" }
```
Render -> Logs: find `event=model_loaded`, `/health status=200`, `prediction_success`, `validation_error` + `status=400`, and `status=404`. Compare request IDs and durations.

### Observation table
| Test | Result | Log evidence |
|---|---|---|
| Startup | service starts | event=model_loaded |
| /health | 200 | INFO http_request status=200 |
| Valid /predict | 200 | prediction_request, prediction_success |
| Invalid /predict | 400 | validation_error, WARNING status=400 |
| Unknown route | 404 | WARNING status=404 |
| Docker | healthy | same events in `docker logs` |
| Actions / Render | green / Live | logs visible in Render |

---

## Troubleshooting
- `unauthorized` on push: re-run `docker login`; token needs `write:packages`. In Actions: set workflow permissions (see P9).
- Render "Image pull failed": package is private or the URL/tag is wrong; make it Public or add a read:packages credential.
- Render 502/no bound port: set `PORT=5000` (the app binds 0.0.0.0 and reads PORT).
- Old behaviour on Render: you forgot Manual Deploy -> Deploy latest reference.
- `git push` rejected: `git pull --rebase origin main`, then push.
- Port 5000 busy locally: use `-p 5001:5000` and `localhost:5001`.
