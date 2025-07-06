# Makefile for the Lung CT Image Generation project
# This Makefile automates common development tasks for macOS/Linux environments.

.PHONY: all venv activate install run test docs clean

# --- Configuration Variables ---
# Directory for the Python virtual environment
VENV_DIR = venv
# Path to the Python executable within the virtual environment
PYTHON = $(VENV_DIR)/bin/python
# Path to the pip executable within the virtual environment
PIP = $(VENV_DIR)/bin/pip
# Path to the main source code directory (quoted for spaces in name)
SRC_DIR = "Synthetic Image Generator"
# Path to the tests directory
TESTS_DIR = "$(SRC_DIR)/tests"
# Path to the documentation directory
DOCS_DIR = docs
# Path to the requirements file
REQUIREMENTS_FILE = requirements.txt

# --- Default Target ---
# Running 'make' without any arguments will execute the 'run' target.
all: run

# --- Virtual Environment Management ---
# Creates a Python virtual environment if it doesn't already exist.
venv:
	@echo "Setting up virtual environment in $(VENV_DIR)..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created at $(VENV_DIR)"; \
	else \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	fi

# Provides instructions on how to activate the virtual environment.
activate: venv
	@echo "To activate the virtual environment, run: source $(VENV_DIR)/bin/activate"

# --- Dependency Management ---
# Installs all required Python packages listed in requirements.txt into the virtual environment.
install: venv
	@echo "Installing Python dependencies from $(REQUIREMENTS_FILE)..."
	@$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "Dependencies installed."

# --- Application Execution ---
# Runs the main application script (main.py) using the virtual environment's Python.
run: install
	@echo "Running the main application..."
	@$(PYTHON) "$(SRC_DIR)/main.py"
	@echo "Application finished."

# --- Testing ---
# Runs the unit tests using Python's unittest discovery.
test: install
	@echo "Running unit tests from $(TESTS_DIR)..."
	@$(PYTHON) -m unittest discover "$(TESTS_DIR)"
	@echo "Tests finished."

# --- Documentation ---
# Builds the Sphinx documentation by calling the Makefile within the docs directory.
docs:
	@echo "Building Sphinx documentation..."
	@$(MAKE) -C $(DOCS_DIR) html
	@echo "Documentation built in $(DOCS_DIR)/build/html"

# --- Cleanup ---
# Cleans up various generated files and directories to keep the project tidy.
clean:
	@echo "Cleaning up project files..."
	# Remove Python bytecode and cache directories
	@find . -name "__pycache__" -type d -exec rm -rf {} + || true
	@find . -name "*.pyc" -delete || true
	@find . -name "*.orig" -delete || true
	# Remove pytest cache (if any, as mentioned in README)
	@rm -rf .pytest_cache || true
	# Remove coverage reports (if any, as mentioned in README)
	@rm -rf .coverage htmlcov/ || true
	# Clean Sphinx documentation build artifacts by calling docs/Makefile's clean target
	@$(MAKE) -C $(DOCS_DIR) clean || true
	# Remove virtual environment directory (uncomment the line below if you want to remove it with 'make clean')
	# @rm -rf $(VENV_DIR) || true
	@echo "Cleanup complete."