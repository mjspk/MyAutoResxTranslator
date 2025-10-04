# MyAutoResxTranslator

MyAutoResxTranslator is a graphical user interface (GUI) application for automatically translating `.resx` files using the DeepL API. It's built with Python and `tkinter`.

This tool is designed to streamline the localization process of .NET applications by providing a simple interface to translate resource files into multiple languages.

## Features

*   **DeepL Integration**: Utilizes the DeepL API for high-quality machine translation.
*   **Protected Words**: Specify a list of words that should not be translated.
*   **RTL Language Support**: Correctly handles right-to-left (RTL) languages.
*   **Glossary Support**: Use a glossary to ensure consistent translation of specific terms.
*   **Translation Preview**: Preview translations before saving the translated `.resx` files.
*   **Bulk Translation**: Translate a single `.resx` file into multiple languages at once.
*   **User-Friendly Interface**: A simple and intuitive GUI for easy operation.

## Installation & Usage

There are two ways to use this application:

### 1. Run from Source

If you have Python installed, you can run the application directly from the source code.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MyAutoResxTranslator.git
    cd MyAutoResxTranslator
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

### 2. Download the Executable

If you prefer not to run the application from the source, you can download a pre-built executable for Windows.

1.  Go to the [**Releases**](https://github.com/your-username/MyAutoResxTranslator/releases) page of this repository.
2.  Download the latest `MyAutoResxTranslator-windows.zip` file.
3.  Unzip the file and run the `MyAutoResxTranslator.exe` executable.

After starting the application, follow these steps:

1.  **Enter your DeepL API Key** in the designated field.
2.  **Click "Browse..."** to select your base `.resx` file.
3.  **Click "Load languages"** to fetch the available target languages from DeepL.
4.  **Select the target languages** you want to translate your file into.
5.  **(Optional)** Add any protected words (comma-separated) that you don't want to be translated.
6.  **(Optional)** Load a glossary file (CSV or JSON) for consistent translations.
7.  **Click "Start translation (with preview)"** to begin the translation process.
8.  A preview window will appear, allowing you to review and edit the translations before saving.
9.  **Click "Save"** in the preview window to save the translated `.resx` files. The new files will be created in the same directory as the original file.

## Credits

This project was created and is maintained by **[mjspk](https://github.com/mjspk)**.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
