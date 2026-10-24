# Upload and submit the project

Repository: https://github.com/DimitriAntypas/unknown-dataset-hackathon

## 1. Update the GitHub repository

1. Extract `Hackathon_Submission_Final.zip` on your computer.
2. Open the repository linked above and sign in to the account that owns it.
3. Keep the repository public.
4. Choose **Add file → Upload files**.
5. Upload the extracted files and folders, not the ZIP itself and not an enclosing folder.
   `main.py`, `supervisor.py`, `requirements.txt`, `README.md`, and the notebook must appear at the root.
6. Include `tests` and `.github`. If hidden files are skipped, create `.github/workflows/ci.yml`
   using **Add file → Create new file**, then paste its supplied contents.
   Drag the entire `tests` folder into the upload area so its scripts stay under `tests/`.
   Keep the workflow at `.github/workflows/ci.yml`; a `ci.yml` file at the repository root is not a workflow.
7. Commit the upload so the repository contains the final README and notebook in this ZIP.
8. Open that URL in a signed-out/private window to confirm the files are public.

## 2. Optionally upload the Colab notebook

1. Open https://colab.research.google.com/ and sign in.
2. Choose **File → Upload notebook** (or the **Upload** tab in the opening dialog).
3. Select `Unknown_Dataset_Hackathon.ipynb` from the extracted files.
4. Use **File → Save a copy in Drive** if needed to retain your own copy.
5. Use a CPU runtime. Run setup and demonstration cells, then optionally the tests.
6. For your own dataset, follow the upload and configuration cells. The notebook already contains the program.
7. Click **Share**. Give the professor viewer access, or enable **Anyone with the link → Viewer** if that is the requested sharing format.
8. Copy the shared notebook link. Check that your intended recipient can open it.

## 3. Check README.md

1. Open README.md in the GitHub repository.
2. Check that its GitHub repository field is https://github.com/DimitriAntypas/unknown-dataset-hackathon.
3. If separately requested, add your shared Colab URL to the optional notebook field and commit that edit.
4. Confirm that the team names match the team approved by the professor.

A Colab link is copied from your saved notebook's Share dialog.

## 4. Submit

Submit https://github.com/DimitriAntypas/unknown-dataset-hackathon as required by the project brief.
The Colab notebook is an optional convenience unless separately requested by the professor.
To obtain a ZIP reflecting the repository, use GitHub's **Code → Download ZIP** after committing your updates.
Follow your professor's actual deadline and submission location. This prepared file package does not itself
change GitHub or submit the project to the professor.

## Final checks

- README contains the real GitHub repository link and the approved team names.
- The brief specifies three students per team; obtain an exception or update the four supplied names.
- GitHub root includes main.py, supervisor.py and requirements.txt.
- If using Colab, setup and demo complete successfully in your runtime.
- If GitHub Actions is enabled, inspect its test result.
- Public example tests do not establish compatibility with any later official input/output specification.
