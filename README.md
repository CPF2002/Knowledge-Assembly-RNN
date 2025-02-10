# Knowledge Assembly RNN

Stephanie Nelli and Compton French

Continuing on the ideas presented in Sheahan et al. 2021 and Nelli et al. 2023. Building a Recurrent Neural Network that will conduct Knowledge Assembly in various training curriculum such as blocked, baseline, and interleaved. The full context range is 1-8 objects, divided into two seperate contexts, 1-4 and 5-8.

## Table of Contents

- [Knowledge Assembly RNN](#knowledge-assembly-rnn)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Cloning the Repository](#cloning-the-repository)
    - [Setup the Repository](#setup-the-repository)
    - [Installing Python Project Dependencies](#installing-python-project-dependencies)
  - [Usage](#usage)

## Installation

### Cloning the Repository

To clone the repository, run the following command:

```bash
git clone https://github.com/CPF2002/Knowledge-Assembly-RNN.git
```

This will create a directory named `Knowledge-Assembly-RNN` in your current working directory containing all the files from the repository.

### Setup the Repository

To set up the repository, create the following folders in the project directory:

```bash
mkdir animations datasets figures models network_analysis network_analysis/lesion_tests network_analysis/RDMs results results/runs trainingrecords trials
```

### Installing Python Project Dependencies

This repository uses several Python packages to support its functionality. 

1. Ensure Python and pip installed. You can check by running:

    ```bash
    # Check Python installation
    python --version #or
    python3 --version

    # Check pip installation
    pip --version #or
    pip3 --version
    ```

    If Python or pip are not installed, follow the instructions for your operating system to install them. The repository is developed on Python v3.12.8. Using a different version of Python may result in some issues.

2. Create a virtual environment:

    ```bash
    # Create a virtual environment
    python -m venv venv
    ```

3. Activate the virtual environment:

    ```bash
    # On Windows
    .\venv\Scripts\activate

    # On Unix or MacOS
    source venv/bin/activate
    ```

4. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

5. Verify installation of packages:

    ```bash
    pip list
    ```

    Below is a list of all dependencies with their respective versions.

    | Package                   | Version    |
    |---------------------------|------------|
    | `absl-py`                 | 2.1.0      |
    | `contourpy`               | 1.2.1      |
    | `cycler`                  | 0.12.1     |
    | `filelock`                | 3.15.1     |
    | `fonttools`               | 4.53.0     |
    | `fsspec`                  | 2024.6.0   |
    | `grpcio`                  | 1.64.1     |
    | `Jinja2`                  | 3.1.4      |
    | `joblib`                  | 1.4.2      |
    | `kiwisolver`              | 1.4.5      |
    | `Markdown`                | 3.6        |
    | `MarkupSafe`              | 2.1.5      |
    | `matplotlib`              | 3.9.0      |
    | `mpmath`                  | 1.3.0      |
    | `networkx`                | 3.3        |
    | `numpy`                   | 1.26.4     |
    | `packaging`               | 24.1       |
    | `pandas`                  | 2.2.2      |
    | `pillow`                  | 10.3.0     |
    | `pip`                     | 24.0       |
    | `protobuf`                | 4.25.3     |
    | `pyparsing`               | 3.1.2      |
    | `python-dateutil`         | 2.9.0.post0|
    | `pytz`                    | 2024.1     |
    | `scikit-learn`            | 1.5.0      |
    | `scipy`                   | 1.13.1     |
    | `setuptools`              | 70.0.0     |
    | `six`                     | 1.16.0     |
    | `sympy`                   | 1.12.1     |
    | `tensorboard`             | 2.17.0     |
    | `tensorboard-data-server` | 0.7.2      |
    | `threadpoolctl`           | 3.5.0      |
    | `torch`                   | 2.2.2      |
    | `torch-tb-profiler`       | 0.4.3      |
    | `torchvision`             | 0.17.2     |
    | `typing_extensions`       | 4.12.2     |
    | `tzdata`                  | 2024.1     |
    | `Werkzeug`                | 3.0.3      |

## Usage

To run the main script from the project directiory, use the following command with your Python version:

```bash
python main.py
```

If you are using Python 3, you may need to use:

```bash
python3 main.py
```

To ensure using Python 3.12, you may need to use:

```bash
python3.12 main.py
```

Ensure that you have activated your virtual environment before running the script.
