"""Write SHA256SUMS for the files the app loads. Run after train.py. Commit the result.
Anyone can re-check with:  python make_manifest.py --check"""
import sys
import security

FILES = ["models/bundle.joblib", "demo_data/ehr_demo.csv", "demo_data/sensor_demo.csv"]
if "--check" in sys.argv:
    for f in FILES:
        security.verify(f)
        print("OK ", f)
else:
    security.MANIFEST.write_text("".join(f"{security.sha256(security.ROOT / f)}  {f}\n" for f in FILES))
    print(security.MANIFEST.read_text())
