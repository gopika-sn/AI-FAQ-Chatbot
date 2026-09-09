import os
import shutil
import zipfile

def build_submission_package():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    submission_dir = os.path.join(base_dir, "submission")
    zip_filepath = os.path.join(base_dir, "AI-FAQ-Chatbot-Submission.zip")

    # Clean existing submission directory & zip if present
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    if os.path.exists(zip_filepath):
        os.remove(zip_filepath)

    os.makedirs(submission_dir, exist_ok=True)

    # Files to include directly in root
    root_files = [
        "app.py",
        "nlp_engine.py",
        "gemini_fallback.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        ".env.example"
    ]

    for fname in root_files:
        src = os.path.join(base_dir, fname)
        dst = os.path.join(submission_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied root file: {fname}")

    # Directories to copy
    copy_dirs = ["data", "tests", "docs"]
    for dname in copy_dirs:
        src_d = os.path.join(base_dir, dname)
        dst_d = os.path.join(submission_dir, dname)
        if os.path.exists(src_d):
            shutil.copytree(src_d, dst_d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
            print(f"Copied directory: {dname}")

    # Create ZIP archive
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, submission_dir)
                zipf.write(abs_file, arcname=os.path.join("AI-FAQ-Chatbot-Submission", rel_path))

    print(f"\nSubmission directory created: {submission_dir}")
    print(f"ZIP package generated successfully: {zip_filepath}")
    print(f"ZIP size: {os.path.getsize(zip_filepath)} bytes")

if __name__ == "__main__":
    build_submission_package()
