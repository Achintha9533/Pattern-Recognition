Project Dependencies
====================

This project relies on several Python packages for its core functionality and for building its comprehensive documentation. It has been tested with Python 3.9 and newer versions.

Core Project Dependencies
-------------------------

The main application dependencies are listed in the `requirements.txt` file located in the root directory of the repository. These packages are essential for running the image generation, training, and evaluation workflows.

To install them, ensure your Python virtual environment is activated and run:

.. code-block:: bash

   pip install -r requirements.txt

Key dependencies include:
* **PyTorch:** The primary deep learning framework for model definition, training, and inference.
* **NumPy:** For numerical operations, especially with image data.
* **Pydicom:** For handling DICOM medical image files.
* **Scikit-image:** For image processing tasks.
* **tqdm:** For displaying progress bars during long operations.
* **gdown:** For downloading pre-trained model weights from Google Drive.
* **torch_fidelity:** For calculating the Fréchet Inception Distance (FID) metric.

*(You might want to expand this list with brief descriptions of other critical packages from your `requirements.txt`.)*

Documentation Dependencies
--------------------------

The packages required to build this documentation (using Sphinx) are listed in the `requirements-docs.txt` file, also located in the root directory of the repository.

To install these, with your Python virtual environment activated, run:

.. code-block:: bash

   pip install -r requirements-docs.txt

Key documentation dependencies include:
* **Sphinx:** The documentation generator itself.
* **sphinx-rtd-theme:** The Read the Docs theme used for the documentation's visual style.
* **myst-parser:** Allows for writing documentation in Markdown (`.md`) in addition to reStructuredText (`.rst`).

*(You might want to expand this list with brief descriptions of other documentation packages if you use more.)*

---

**Note on PyTorch Installation:**

The `requirements.txt` file typically specifies the CPU version of PyTorch by default. If you have a CUDA-enabled GPU and wish to leverage it for faster training and inference, please consult the `official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ to find the exact installation command that matches your CUDA version and operating system.