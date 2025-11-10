<h1 align="center">SUP Toolbox App 🎨</h1>
<p align="center">
  A user-friendly Gradio web interface for the powerful <strong>SUP-Toolbox</strong> image restoration and upscaling library.
</p>

<p align="center">
  <img src="https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/screen_sup_toolbox.PNG" alt="SUP Toolbox App Interface" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/DEVAIEXP/sup-toolbox-app/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>  
  <a href="https://github.com/DEVAIEXP/sup-toolbox/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version"></a>
  <a href="https://github.com/DEVAIEXP/sup-toolbox-app/stargazers"><img src="https://img.shields.io/github/stars/DEVAIEXP/sup-toolbox-app?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/DEVAIEXP/sup-toolbox"><img src="https://img.shields.io/badge/Powered%20by-SUP--Toolbox-orange" alt="Powered by SUP-Toolbox"></a>
  <a href="https://huggingface.co/spaces/elismasilva/sup-toolbox-app"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow" alt="Hugging Face Spaces"></a>
</p>

## About

The **SUP Toolbox App** provides an intuitive and interactive web interface for the [SUP-Toolbox library](https://github.com/DEVAIEXP/sup-toolbox). It allows you to harness the power of advanced image enhancement models like **SUPIR**, **FaithDiff**, and **ControlNetTile** through a user-friendly Gradio application, without needing to use the command line.

This application is designed for artists, photographers, and enthusiasts who want to visually experiment with different settings, manage presets, and process images one by one with real-time feedback.

## Features

-   **Interactive UI:** A clean and organized Gradio interface for easy access to all features.
-   **Full Engine Support:** Access and combine all restoration (`SUPIR`, `FaithDiff`) and upscaling (`SUPIR`, `FaithDiff`, `ControlNetTile`) engines from the core library.
-   **Preset Management:** Create, save, and load your custom workflows as `.json` presets directly from the UI.
-   **Live Previews & Comparison:** Instantly see the results of your settings with before-and-after image sliders.
-   **Metadata Handling:** Automatically saves all generation parameters within the output image. Load settings from a previously generated image by simply dropping it into the app.
-   **Real-time Logging:** A live log panel shows the progress of the image generation process.

## Requirements

This application depends on the `sup-toolbox` library. Therefore, it shares the same hardware and software requirements. For a detailed breakdown, please refer to the core library's documentation:

➡️ **[View Full Requirements in the SUP-Toolbox README](https://github.com/DEVAIEXP/sup-toolbox#requirements)**

In summary:
-   **Python 3.10+**, Git
-   **NVIDIA GPU** with 12GB+ VRAM recommended.
-   **~83 GB** of disk space for models.

## Installation

The setup process is designed to be simple. It will create a virtual environment and install both the `sup-toolbox` library from its GitHub repository and all other necessary Python packages.

### 1. Clone the Application Repository
```bash
git clone https://github.com/DEVAIEXP/sup-toolbox-app.git
cd sup-toolbox-app
```

### 2. Run the Setup Script

Choose the script that matches your operating system and terminal.

*   **On Windows (using PowerShell):**
    This is the recommended method for modern Windows systems.

    1.  **Allow script execution (one-time setup):**
        If you haven't run PowerShell scripts before, open PowerShell **as an Administrator** and run:
        ```powershell
        Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
        ```
        Press `Y` and `Enter` to confirm.

    2.  **Run the script:**
        In a regular PowerShell terminal, run:
        ```powershell
        .\setup.ps1
        ```

*   **On Windows (using Command Prompt):**
    ```batch
    .\setup.bat
    ```

*   **On Linux or macOS:**
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

This will set up everything you need. The `requirements.txt` file in this repository is configured to automatically download and install the correct version of the `sup-toolbox` library.

## Usage

After a successful installation, you can easily start the application using the provided start scripts.

*   **On Windows:** Double-click `start.bat` or `start.ps1`, or run them from your terminal:
    ```powershell
    # In PowerShell
    .\start.ps1
    ```
    ```batch
    :: In Command Prompt
    .\start.bat
    ```

*   **On Linux or macOS:**
    ```bash
    ./start.sh
    ```

This will activate the virtual environment and launch the Gradio application. Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

## Relationship with `sup-toolbox`

This project is the official Gradio UI front-end for the `sup-toolbox` library.

-   **[sup-toolbox (Library)](https://github.com/DEVAIEXP/sup-toolbox):** The core engine. It's a Python library and CLI tool that contains all the processing logic, model management, and configuration system. It's designed for programmatic use and automation.
-   **sup-toolbox-app (This Project):** A user-friendly graphical interface that *uses* the `sup-toolbox` library as its backend. It provides no image processing logic itself but offers a convenient way to access the library's features.

## Releases and Changelog

For a detailed list of changes, new features, and bug fixes for each version of this application, please see the **[CHANGELOG.md](CHANGELOG.md)** file.

## Licensing

The **SUP Toolbox App** (this project) is licensed under the **Apache License, Version 2.0**. You can find the full license text in the `LICENSE` file in this repository.

Please be aware that this application is a front-end for the `sup-toolbox` library, which integrates several third-party components, some of which are under **non-commercial licenses** (such as SUPIR).

**By using this application, you acknowledge and agree to comply with all the licensing terms of the underlying `sup-toolbox` library and its components.** For a complete breakdown of these licenses, please refer to the library's documentation:

➡️ **[View Full Licensing Details in the SUP-Toolbox README](https://github.com/DEVAIEXP/sup-toolbox#licensing)**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DEVAIEXP/sup-toolbox-app&type=Date)](https://star-history.com/#DEVAIEXP/sup-toolbox-app&Date)

## Contact
For questions or issues related to this Gradio application, please open an issue in this repository. For issues related to the core processing logic, please refer to the [`sup-toolbox`](https://github.com/DEVAIEXP/sup-toolbox/issues) library's repository.

Project Contact: contact@devaiexp.com