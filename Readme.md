# Hybrid Retrieval System

## 1. Project Setup

Open the project folder in VS Code and open the PowerShell terminal.

```powershell
cd "C:\Users\sam\Desktop\PE\hybrid retrieval"
```

## 2. Create a Virtual Environment

Run this once:

```powershell
python -m venv venv
```

## 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of the terminal prompt.

## 4. Install Dependencies

Install all dependencies listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Optional: upgrade pip if you encounter package-installation issues.

```powershell
python -m pip install --upgrade pip
```

If a required package is still missing, install it in the activated environment. For example:

```powershell
python -m pip install langchain-community pypdf
python -m pip install pydantic pydantic-settings
python -m pip install streamlit
```

## 5. Select the Correct Python Interpreter in VS Code

1. Press `Ctrl + Shift + P`.
2. Select **Python: Select Interpreter**.
3. Choose the interpreter at:

```text
C:\Users\sam\Desktop\PE\hybrid retrieval\venv\Scripts\python.exe
```

If Pylance still reports missing imports, run **Developer: Reload Window** from the Command Palette.

## 6. Run the Application

The project appears to use Streamlit. Start it with:

```powershell
streamlit run app.py
```

Then open the local URL printed in the terminal. The usual default is:

```text
http://localhost:8501
```

If port 8501 is already in use:

```powershell
streamlit run app.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## 7. Quick Start for Future Runs

After the initial setup, use these commands whenever you reopen the project:

```powershell
cd "C:\Users\sam\Desktop\PE\hybrid retrieval"
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## 8. Verify Key Dependencies

Run these commands inside the activated virtual environment:

```powershell
python -c "import streamlit; print('Streamlit working:', streamlit.__version__)"
python -c "import pydantic; print('Pydantic working:', pydantic.__version__)"
python -c "from langchain_community.document_loaders import PyPDFLoader; print('PyPDFLoader working')"
```

## 9. Troubleshooting

### `ModuleNotFoundError`

If Python reports that a module is missing, install the corresponding package in the activated virtual environment:

```powershell
python -m pip install package-name
```

Then run the application again.

### Pylance: `reportMissingImports`

Confirm that VS Code is using the project's `venv\Scripts\python.exe` interpreter (see Section 5).

### Check which Python is active

```powershell
where.exe python
python --version
python -m pip --version
```

The first Python path should point to this project's `venv` folder.

## 10. Important Notes

- Activate the virtual environment before installing packages or running the application.
- Keep `storage\chroma_db` and `storage\docstore` intact; they contain the project's existing indexed data.
- Do not delete or recreate the storage folders unless you intentionally want to rebuild the indexes.
- If `requirements.txt` does not include a package that the code imports, add the appropriate dependency to that file after confirming the package name.

---

**Project folder:** `C:\Users\sam\Desktop\PE\hybrid retrieval`
