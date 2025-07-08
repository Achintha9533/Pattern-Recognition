Installation & Setup
====================

This project has been tested with Python 3.9+ on macOS and Windows, and should work with compatible PyTorch versions.

See :doc:`requirements` for the full list of dependencies. *(Note: I've assumed you might create a requirements.rst, or you can directly reference the file on GitHub here)*

To install the application, you can just clone this repository and use pip.

1.  **Clone the repository:**

    .. code-block:: bash

        git clone https://github.com/Achintha9533/Pattern-Recognition.git
        cd Pattern-Recognition

2.  **Create and activate a virtual environment (recommended):**

    On macOS/Linux:

    .. code-block:: bash

        python -m venv venv
        source venv/bin/activate

    On Windows:

    .. code-block:: bash

        python -m venv venv
        .\venv\Scripts\activate

3.  **Install dependencies:**

    All required Python packages are listed in `requirements.txt`. With your virtual environment activated, install them using pip:

    .. code-block:: bash

        pip install -r requirements.txt

    .. note::
       **Note on PyTorch:** The `requirements.txt` will install the CPU version of PyTorch by default (or the specific version frozen). If you have a CUDA-enabled GPU and wish to utilize it, please refer to the `official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for the exact command that matches your CUDA version.

4.  **Data Setup:**

    This project requires a Lung CT DICOM dataset. The code is configured to expect patient subfolders named `QIN LUNG CT 1` through `QIN LUNG CT 47`.

    * Download your dataset (e.g., the QIN LUNG CT dataset if you have access to it).
    * Place the root directory of your dataset (e.g., the folder containing `QIN LUNG CT 1`, etc.) in a convenient location on your system.
    * **IMPORTANT:** Update the `BASE_DIR` variable in `synthetic_image_generator/config.py` to the **absolute path** of your dataset's root directory.

    .. code-block:: python

        # synthetic_image_generator/config.py
        from pathlib import Path
        BASE_DIR = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT") # <-- Location