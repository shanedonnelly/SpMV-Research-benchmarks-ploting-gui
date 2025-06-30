# SpMV-Research-benchmarks-ploting-gui
## Description
This repository provides a fully functional [Streamlit](https://streamlit.io/) app to visualize results from the [SpMV-Research-benchmarks](https://github.com/pmpakos/SpMV-Research-benchmarks). The app allows users to filter CSV files, save them as subsets, and generate boxplots from it. You can choose one Y-axis dimension to plot, and up to two X-axis dimensions. 

The app includes several features around plotting, such as:
- customizing the plot’s title, size, labels, and image format  
- plotting multiple subsets together  
- ...

This app was developed by my self as part of an internship project at [CSLab](http://www.cslab.ntua.gr/). The goal was to create an internal tool for researchers to save time when generating graphical plots.

## Installation
To download the app, simply clone this repository : 
```bash
git clone https://github.com/shanedonnelly/SpMV-Research-benchmarks-ploting-gui.git
cd SpMV-Research-benchmarks-ploting-gui/
```
## Usage
To use it, first copy your CSV files into `app/csv`  or if you want, you can create a symlink with : 
```bash
ln -s /path/to/file app/csv/synthetic_benchmarks_all-devices_all.csv #Rename with the corresponding file name
```
You must put the csv files when the app is not running for keeping the app simple. 
Then, to run, you can either use of the two running scripts `python_venv_run.sh` and `python_venv_run_clean.sh` : 

```bash
chmod +x python_venv_run.sh python_venv_run_clean.sh
```
Then, 
```bash 
./python_venv_run.sh
```
or 
```bash 
./python_venv_run_clean.sh #Wich auto-remove all python dependencies each time to save storage
```