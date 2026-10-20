# Security and privacy

**Honest summary:** no software can promise "zero breaches". This project reduces risk by (1) never holding real patient
data, (2) refusing tampered files, (3) cleaning every input, and (4) keeping the dashboard private by default. If you ever
use real patient data, this prototype is **not** enough (see the last section).

## What is in place
| Risk | Protection | Where |
|---|---|---|
| Real patient data leaking | There is none. Data is synthetic, every row tagged `synthetic`, and loaders refuse anything else | `security.require_synthetic`, `train.py` |
| A swapped or poisoned model file (saved models can run code when loaded) | Model and demo data load only if their SHA-256 hash matches `SHA256SUMS`; files outside the project are refused | `security.verify` |
| Malicious or broken sensor input | Every reading is type-checked, range-checked and whitelisted; bad values become "unknown" | `security.sanitise`, `sanitise_ehr` |
| Strangers opening the dashboard | Listens on `127.0.0.1` only. Optional password: set `TWINHEALTH_PASSWORD` (5 wrong tries lock the session, constant-time compare) | `.streamlit/config.toml`, `security.password_ok` |
| Web attacks | XSRF protection on, upload size 1 MB (the app has no uploads), developer toolbar hidden, errors show no file paths | `.streamlit/config.toml` |
| Data sent out | No usage statistics, no external API calls, no secrets in the repo | config, `.gitignore`, secret scan |
| Vulnerable libraries | Exact versions pinned; `pip-audit` (clean when run) and weekly Dependabot | `requirements.txt`, `.github/` |
| Insecure code | `bandit` scan in CI (no medium or high findings when run) | `.github/workflows/ci.yml` |
| Accidentally committing big data or secrets | `.gitignore` blocks `data/`, `.env`, `secrets.toml` | `.gitignore` |

## How to run it privately
```
export TWINHEALTH_PASSWORD="choose-a-long-passphrase"   # optional
streamlit run app.py                                    # opens on this computer only
```
If you deploy it online, put it behind HTTPS and real sign-in (for example your host's authentication). A single shared password is not enough for the public internet.

## Limits (not covered)
- Hash checks catch a changed file, but anyone who can edit both the file and `SHA256SUMS` can defeat them. Commit `SHA256SUMS` and protect your repository (two-factor sign-in, branch protection).
- Python model files are a pickle format. Only load model files you built yourself.
- There is no user accounts system, audit log or encryption at rest.

## If you ever use REAL patient data (not allowed in this competition)
You would need, at least: legal basis and consent under the DPDP Act (and HIPAA if US data), data minimisation and de-identification, encryption in transit and at rest, role-based access, audit logs, a breach-response plan, secure hosting, and a security review. None of that is built here.

## Reporting a problem
Open a private security issue on the repository, or contact the team listed in `README.md`.
