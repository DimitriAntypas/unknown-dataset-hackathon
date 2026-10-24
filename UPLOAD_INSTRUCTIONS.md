# Upload and submit the project

## 1. Create the GitHub repository

1. Extract `Hackathon_Submission.zip` on your computer.
2. Sign in at https://github.com and create a new repository named `unknown-dataset-hackathon`.
3. Choose **Public**. The package already supplies README.md, so do not initialize another README.
4. On the new repository page, choose **uploading an existing file**, or **Add file → Upload files**.
5. Upload the extracted files and folders, not the ZIP itself and not an enclosing folder.
   `main.py`, `supervisor.py`, `requirements.txt`, `README.md`, and the notebook must appear at the root.
6. Include `tests` and `.github`. If hidden files are skipped, create `.github/workflows/ci.yml`
   using **Add file → Create new file**, then paste its supplied contents.
7. Commit the upload. Copy the repository page URL from your address bar.
8. Open that URL in a signed-out/private window to confirm the files are public.

## 2. Upload the Colab notebook

1. Open https://colab.research.google.com/ and sign in.
2. Choose **File → Upload notebook** (or the **Upload** tab in the opening dialog).
3. Select `Unknown_Dataset_Hackathon.ipynb` from the extracted files.
4. Use **File → Save a copy in Drive** if needed to retain your own copy.
5. Use a CPU runtime. Run setup and demonstration cells, then optionally the tests.
6. For your own dataset, follow the upload and configuration cells. The notebook already contains the program.
7. Click **Share**. Give the professor viewer access, or enable **Anyone with the link → Viewer** if that is the requested sharing format.
8. Copy the shared notebook link. Check that your intended recipient can open it.

## 3. Put the links in README.md

1. Open README.md in the GitHub repository and click Edit.
2. Replace `REPLACE_WITH_YOUR_PUBLIC_GITHUB_REPOSITORY_URL` with the real repository URL.
3. Optionally replace `REPLACE_WITH_YOUR_SHARED_COLAB_NOTEBOOK_URL` with a shared Colab URL.
4. Commit the updated README.
5. Download the updated README for any file-based submission; do not submit the placeholder version.

A valid GitHub link has the shape `https://github.com/YOUR_USERNAME/unknown-dataset-hackathon`.
A Colab link is copied from your saved notebook's Share dialog. Do not invent either URL.

## 4. Submit

The project brief requires the public GitHub repository URL as the submission.
The Colab notebook is an optional convenience unless separately requested by the professor.
To obtain a ZIP reflecting the final README, use GitHub's **Code → Download ZIP** after committing your links.
Follow your professor's actual deadline and submission location. Creating these accounts/uploads is separate
from this prepared file package; no upload or final submission has yet been made on your behalf.

## Final checks

- README contains the real GitHub repository link and the approved team names.
- The brief specifies three students per team; obtain an exception or update the four supplied names.
- GitHub root includes main.py, supervisor.py and requirements.txt.
- Colab setup and demo complete successfully in your runtime.
- If GitHub Actions is enabled, inspect its test result.
- Public example tests do not establish compatibility with any later official input/output specification.
