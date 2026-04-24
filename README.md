# Urban-pulse-nyc

### Datasets

* [NYPD Complaint Data Historic](https://drive.google.com/file/d/1c1P43ba6sPn_11zNGfYZd1q2o2SjS_1b/view?usp=sharing)
* [311 Service Requests from 2020 to Present](https://drive.google.com/file/d/1OM8vJCHGGApUt8C6gGvEarbQKD39B-FH/view?usp=sharing)
* **Setup**: Please download these datasets and store them within your local repository folder for analysis.

## Preprocessing & Data Aggregation

Before running any notebooks or models, you **must** run the preprocessing script to generate the aggregated dataset:
```bash
python preprocess_data.py
```
**What this does:**
1. Parses the massive raw datasets.
2. Extracts the `YYYY-MM` from the raw date strings.
3. Aggregates all data by both `Police Precinct` AND `YearMonth` to create a robust Monthly Time-Series dataset.
4. Generates a lightweight `preprocessed_data.csv` (4621 rows) that is used to train our Random Forest Regressor model.

## Project Structure & Setup Files

To ensure everything runs smoothly across all operating systems, our repository includes several important configuration files:
* **`requirements.txt`**: This file contains all the Python libraries (pandas, scikit-learn, seaborn, etc.) needed to run our models. You can install them all at once by running `pip install -r requirements.txt` in your terminal.
* **`.gitignore`** (and `.dockerignore`): These files prevent Git and Docker from accidentally trying to process our raw datasets, temporary Python files, and the `output/` folder. They keep our repository fast and lightweight.

## Contribution Guidelines

To keep the project organized and prevent merge conflicts, please follow this workflow:
* **No Direct Commits to `main`**: Always work on a separate branch.
* **Merging**: Once your program is tested and working, merge your branch into `main`.
* **Syncing**: Remember to `git pull` from `main` frequently to stay up to date with other group memebers.
