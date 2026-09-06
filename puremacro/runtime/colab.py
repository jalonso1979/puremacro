"""Juno / Pyodide to Google Colab task offloading bridge.

Allows users working on an iPad (Juno) or in browser environments (Pyodide)
to export heavy compute jobs (MCMC chains, large bootstrap replications,
high-dimensional FAVAR, VFI grid searches) into self-contained Google Colab
notebooks with built-in Google account authentication and Google Drive syncing.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def colab_auth_guide(
    account_email: str | None = None,
    drive_folder: str = "puremacro_jobs",
) -> str:
    """Return step-by-step instructions for Google Colab authentication on iPad/mobile."""
    target_acct = f" ({account_email})" if account_email else ""
    return (
        f"Google Colab Authentication & Offloading Guide for iPad / Juno:\n"
        f"===============================================================\n"
        f"1. Open the generated notebook in Safari or Google Chrome by navigating to:\n"
        f"   https://colab.research.google.com\n"
        f"2. Tap 'Upload' and select your exported .ipynb notebook.\n"
        f"3. When prompted in the notebook, run the authentication cell:\n"
        f"   >>> from google.colab import auth\n"
        f"   >>> auth.authenticate_user()\n"
        f"   A Google sign-in modal will pop up. Select your Google account{target_acct} and grant access.\n"
        f"   *Tip for iPad/Juno*: If Safari blocks popups, tap the URL bar icon to allow popups for colab.research.google.com.\n"
        f"4. (Optional) To persist results directly to your Google Drive, run:\n"
        f"   >>> from google.colab import drive\n"
        f"   >>> drive.mount('/content/drive')\n"
        f"   Results will automatically sync to: Google Drive / MyDrive / {drive_folder} /\n"
        f"5. Execute the computation cells. Colab's cloud CPUs/GPUs will run the task.\n"
        f"6. The notebook saves results as a portable .pmz cartridge that syncs back\n"
        f"   to your iPad or local machine via Google Drive or direct download."
    )


def colab_auth_snippet(
    *,
    mount_drive: bool = True,
    drive_folder: str = "puremacro_jobs",
    require_secrets: Sequence[str] | None = None,
) -> str:
    """Generate python code snippet for Google Colab authentication and secret retrieval."""
    lines = [
        "# Google Account Authentication & Drive Mount",
        "try:",
        "    from google.colab import auth, drive",
        "    import os, shutil",
        "    print('Authenticating Google account...')",
        "    auth.authenticate_user()",
    ]
    if mount_drive:
        lines += [
            "    print('Mounting Google Drive at /content/drive...')",
            "    drive.mount('/content/drive')",
            f"    drive_dir = '/content/drive/MyDrive/{drive_folder}'",
            "    os.makedirs(drive_dir, exist_ok=True)",
            "    print(f'Sync directory ready: {drive_dir}')",
        ]
    lines += [
        "    print('Authentication and Drive setup complete.')",
        "except Exception as e:",
        "    print(f'Drive mount skipped or running outside Colab: {e}')",
    ]

    if require_secrets:
        lines += [
            "",
            "# Retrieve credentials from Colab Secret Manager or prompt interactively",
            "import getpass",
            "try:",
            "    from google.colab import userdata",
            "except ImportError:",
            "    userdata = None",
            "",
        ]
        for sec in require_secrets:
            lines += [
                f"# Secret: {sec}",
                f"{sec} = None",
                "if userdata is not None:",
                "    try:",
                f"        {sec} = userdata.get('{sec}')",
                "    except Exception:",
                "        pass",
                f"if not {sec}:",
                f"    {sec} = getpass.getpass('Please enter your {sec}: ')",
            ]

    return "\n".join(lines)


def colab_badge(notebook_url_or_repo: str, branch: str = "main") -> str:
    """Generate a Markdown badge linking to Google Colab.

    Parameters
    ----------
    notebook_url_or_repo : str
        Either a full GitHub notebook URL or 'owner/repo/blob/branch/path.ipynb'.
    branch : str, default 'main'
        Branch name if specifying a repo relative path.

    Returns
    -------
    str
        Markdown string with the official Google Colab badge.
    """
    clean_url = notebook_url_or_repo.strip()
    if clean_url.startswith("https://github.com/"):
        rel_part = clean_url[len("https://github.com/") :]
        colab_link = f"https://colab.research.google.com/github/{rel_part}"
    elif "/" in clean_url and not clean_url.startswith("http"):
        colab_link = f"https://colab.research.google.com/github/{clean_url}"
    else:
        colab_link = clean_url

    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})"


def generate_colab_notebook(
    code: str,
    *,
    data_payloads: Mapping[str, Any] | None = None,
    title: str = "puremacro Cloud Compute Task",
    save_path: str | Path | None = "colab_task.ipynb",
    mount_drive: bool = True,
    drive_folder: str = "puremacro_jobs",
    use_accelerator: bool = False,
    pip_extras: Sequence[str] | None = None,
    require_secrets: Sequence[str] | None = None,
    output_filename: str | None = "colab_results.pmz",
) -> dict:
    """Generate a self-contained Jupyter notebook tailored for execution in Google Colab.

    Parameters
    ----------
    code : str
        Python code to execute in Google Colab.
    data_payloads : dict, optional
        Mapping of filename -> DataFrame or ndarray. These datasets will be
        embedded directly into the notebook as base64-encoded strings, so the
        Colab session runs 100% offline without manual file uploads.
    title : str, default "puremacro Cloud Compute Task"
        Title for the Colab notebook.
    save_path : str or Path, optional
        Path where the .ipynb file will be saved on disk.
    mount_drive : bool, default True
        If True, includes cells to authenticate and mount Google Drive.
    drive_folder : str, default "puremacro_jobs"
        Google Drive folder name where output artifacts will be saved.
    use_accelerator : bool, default False
        If True, configures the notebook to install numba/accelerator extras.
    pip_extras : sequence of str, optional
        Additional pip extras, e.g. ["backend", "dev"].
    require_secrets : sequence of str, optional
        Names of required secrets/API tokens (e.g. ["FRED_API_KEY"]).
    output_filename : str, optional, default "colab_results.pmz"
        Filename of the output artifact to serialize and export back to the user.

    Returns
    -------
    dict
        Jupyter Notebook structure as a dict.
    """
    cells: list[dict[str, Any]] = []

    # Cell 1: Markdown Introduction & Guide
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "This notebook was generated by **puremacro** to offload heavy macroeconomic computation\n",
            "(long bootstraps, MCMC, high-dimensional factor models, or DSGE solving) to Google Colab.\n",
            "\n",
            "### Instructions:\n",
            "1. Run the setup cell below to install `puremacro`.\n",
            "2. Authenticate your Google account to enable Google Drive storage and output syncing.\n",
            "3. Run the computation cells.\n",
            f"4. Results will automatically download and sync to Google Drive (`/MyDrive/{drive_folder}/`).\n",
        ],
    })

    # Cell 2: Installation
    extras_str = ""
    if pip_extras:
        extras_str = f"[{','.join(pip_extras)}]"
    elif use_accelerator:
        extras_str = "[backend]"

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install puremacro in the Colab cloud environment\n",
            f"!pip install -q 'puremacro{extras_str}'\n",
        ],
    })

    # Cell 3: Google Account Authentication & Drive & Secrets
    if mount_drive or require_secrets:
        snippet = colab_auth_snippet(
            mount_drive=mount_drive,
            drive_folder=drive_folder,
            require_secrets=require_secrets,
        )
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in snippet.splitlines()],
        })

    # Cell 4: Embedded Data Payloads (if any)
    if data_payloads:
        payload_lines = [
            "# 3. Unpack embedded data payloads\n",
            "import io\n",
            "import base64\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
        ]
        for name, obj in data_payloads.items():
            if not str(name).isidentifier():
                raise ValueError(
                    f"data_payloads key {name!r} must be a valid Python identifier: it becomes "
                    "the variable name that holds the object inside the notebook"
                )
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                # Parquet round-trips the index (DatetimeIndex, PeriodIndex, MultiIndex)
                # and dtypes exactly; CSV silently turned the index into a text column.
                frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
                bio = io.BytesIO()
                frame.to_parquet(bio, index=True)
                b64 = base64.b64encode(bio.getvalue()).decode("ascii")
                payload_lines.append(f"# Data: {name} (parquet, index preserved)\n")
                payload_lines.append(f"{name}_b64 = '{b64}'\n")
                payload_lines.append(
                    f"{name} = pd.read_parquet(io.BytesIO(base64.b64decode({name}_b64)))\n"
                )
                if isinstance(obj, pd.Series):
                    payload_lines.append(f"{name} = {name}.iloc[:, 0]\n")
            elif isinstance(obj, np.ndarray):
                bio = io.BytesIO()
                np.save(bio, obj)
                b64 = base64.b64encode(bio.getvalue()).decode("ascii")
                payload_lines.append(f"# Array: {name}\n")
                payload_lines.append(f"{name}_b64 = '{b64}'\n")
                payload_lines.append(
                    f"{name} = np.load(io.BytesIO(base64.b64decode({name}_b64)))\n"
                )
            else:
                raise TypeError(
                    f"data_payloads[{name!r}] is a {type(obj).__name__}; only pandas "
                    "DataFrame/Series and numpy arrays can be embedded"
                )

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": payload_lines,
        })

    # Cell 5: User Code
    code_lines = [line + "\n" for line in code.strip().splitlines()]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines,
    })

    # Cell 6: Completion & Results Export (.pmz cartridge)
    if output_filename:
        export_code = [
            "# 4. Serialize results and sync back to iPad / Juno / Local Session\n",
            "import os, shutil\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "from puremacro.runtime.store import save_frame\n",
            "\n",
            "# Locate result variable\n",
            "out_file = '" + output_filename + "'\n",
            "_res = locals().get('result', locals().get('results', locals().get('df_res', None)))\n",
            "if _res is not None:\n",
            "    if hasattr(_res, 'to_frame'):\n",
            "        save_frame(_res.to_frame(), out_file)\n",
            "    elif hasattr(_res, 'to_dataframe'):\n",
            "        save_frame(_res.to_dataframe(), out_file)\n",
            "    elif isinstance(_res, (pd.DataFrame, pd.Series)):\n",
            "        save_frame(_res if isinstance(_res, pd.DataFrame) else _res.to_frame(), out_file)\n",
            "    else:\n",
            "        import pickle\n",
            "        with open(out_file, 'wb') as _f:\n",
            "            pickle.dump(_res, _f)\n",
            "    print(f'Saved result cartridge to {out_file}.')\n",
            "\n",
            "    # Sync to Google Drive if mounted\n",
            f"    drive_dst = Path('/content/drive/MyDrive/{drive_folder}/{output_filename}')\n",
            "    if drive_dst.parent.exists():\n",
            "        shutil.copy(out_file, drive_dst)\n",
            "        print(f'Synced to Google Drive: {drive_dst}')\n",
            "\n",
            "    # Trigger browser download prompt for mobile / iPad\n",
            "    try:\n",
            "        from google.colab import files\n",
            "        files.download(out_file)\n",
            "        print('Browser download triggered.')\n",
            "    except Exception as _e:\n",
            "        print(f'Download prompt skipped: {_e}')\n",
            "else:\n",
            "    print('Task completed successfully. (Define a variable named `result` to auto-export).')\n",
        ]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": export_code,
        })
    else:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execution finished successfully.\n",
                "print('Task finished successfully on Colab cloud.')\n",
            ],
        })

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": title,
                "provenance": [],
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "cells": cells,
    }

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    return nb


def load_colab_result(path_or_bytes: str | Path | bytes) -> Any:
    """Load a result cartridge generated by a Google Colab task.

    Parameters
    ----------
    path_or_bytes : str, Path, or bytes
        Path to the downloaded result file (.pmz, .parquet, or .pkl),
        or raw bytes.

    Returns
    -------
    pd.DataFrame or object
        The loaded results object.
    """
    from puremacro.runtime.store import load_frame

    if isinstance(path_or_bytes, bytes):
        bio = io.BytesIO(path_or_bytes)
        try:
            return load_frame(bio)
        except Exception:
            bio.seek(0)
            import pickle
            return pickle.load(bio)

    p = Path(path_or_bytes)
    if not p.exists():
        raise FileNotFoundError(f"Result cartridge not found: {p}")

    # Try puremacro portable store first (.pmz / .npz)
    try:
        return load_frame(p)
    except Exception:
        pass
    # A genuine puremacro.pocket cartridge (pocket.pack): return its frame
    try:
        from puremacro import pocket as _pocket
        return _pocket.load(p).frame()
    except Exception:
        pass

    # Try parquet if pyarrow is present
    if p.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(p)

    # Try pickle fallback
    import pickle
    with p.open("rb") as f:
        return pickle.load(f)


def show_colab_offload_dialog(
    notebook_path: str | Path,
    *,
    title: str = "puremacro Task Offloaded to Google Colab",
    drive_folder: str = "puremacro_jobs",
) -> Any:
    """Display interactive offloading instructions in Jupyter/Juno or terminal."""
    p = Path(notebook_path)
    abs_path = p.resolve()

    guide_text = colab_auth_guide(drive_folder=drive_folder)
    in_kernel = False
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        in_kernel = ip is not None and ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        in_kernel = False
    if not in_kernel:
        # Outside a Jupyter/Juno kernel the rich HTML card would only print its
        # repr, so show (and return) the documented plain-text guide instead.
        text = f"\n{title}\n{'=' * len(title)}\nNotebook generated at: {abs_path}\n\n{guide_text}"
        print(text)
        return text

    try:
        from IPython.display import HTML, display

        html_content = f"""
        <div style="border: 2px solid #4285F4; border-radius: 8px; padding: 16px; margin: 10px 0; background: #f8fafd; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 24px; margin-right: 10px;">🚀</span>
                <h3 style="margin: 0; color: #1a73e8;">{title}</h3>
            </div>
            <p style="margin: 4px 0 12px 0; color: #3c4043;">
                Notebook ready at: <code>{abs_path}</code>
            </p>
            <ol style="margin: 0 0 12px 0; padding-left: 20px; color: #3c4043;">
                <li>Open <a href="https://colab.research.google.com" target="_blank" style="color: #1a73e8; font-weight: bold;">Google Colab</a> in Safari or Chrome.</li>
                <li>Upload <code>{p.name}</code>.</li>
                <li>Run the authentication cell and sign in with your Google Account.</li>
                <li>Output cartridge will automatically sync back to <code>MyDrive/{drive_folder}/</code>.</li>
            </ol>
            <a href="https://colab.research.google.com" target="_blank" style="display: inline-block; background-color: #1a73e8; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: 500; font-size: 13px;">
                Open Google Colab ↗
            </a>
        </div>
        """
        html_obj = HTML(html_content)
        display(html_obj)
        return html_obj
    except Exception:
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Notebook generated at: {abs_path}\n")
        print(guide_text)
        return guide_text


__all__ = [
    "colab_auth_guide",
    "colab_auth_snippet",
    "colab_badge",
    "generate_colab_notebook",
    "load_colab_result",
    "show_colab_offload_dialog",
]
