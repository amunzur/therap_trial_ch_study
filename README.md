# Clonal hematopoiesis in TheraP trial

This repository contains python scripts used to generate the figures in our clonal hematopoiesis in TheraP trial study. 
Link:

---

## Installation

You can download all dependencies using the yaml file provided in envs/therap_ch_env.yaml.

```
conda env create -f environment.yml
conda activate therap_ch_env
```
Before running the python scripts, please set the environment variable by `export="path/to/project/directory"`. 

## File Structure

- `plotting_scripts/main`: Generates main figures.
- `plotting_scripts/supp` Generates supplementary figures. 
- `figures`: Contains examples of generated figures.

## Notes
Some plotting scripts involving clinical data contain the following variables: `path_clin_data`, `path_age_df`, `path_weeks_to_progression` and `path_ae`. These cannnot be provided online to protect patient privacy. If you require access to these files, please contact the corresponding author. 
