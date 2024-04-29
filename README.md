# DeepLearningFotballAnalysis

## Setup

To set up and run this project, follow these steps:

1. Install Python 3.10
2. Clone this repository to your local machine.
3. Create a virtual environment:

  ```
  python3.10 -m venv venv 
  ```
or 
  ```
  conda create -n venv --file requirements.txt
  ```

4. Activate the virtual environment:

- On Windows:

  ```
  .\venv\Scripts\activate
  ```

- On macOS/Linux:

  ```
  source venv/bin/activate 
  ```
  or
  ```
   conda activate venv
  ```

5. Install the required packages using `requirements.txt`:

    ```
    pip install -r requirements.txt
    ```

6. If you update the the current packages in the `requirements.txt` use:
    ```
    pip freeze > requirements.txt
    ```
    This command lists all the installed packages in the current environment along with their versions and saves them to `requirements.txt`.

## Usage

To use the project, simply run your Python scripts within the activated virtual environment.

## Link to IDUN guide 
https://github.com/jarlsondre/idun_tutorial/blob/main/tutorial.md
