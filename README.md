# Trying to understand Asylum Practices in Denmark.

This repository contains the source code for the analysis of data from the Danish Refugee Appeals Board (Flygtningenævnet). The data are case summaries in textual format and are publicly available from https://fln.dk/praksis.

The source code for the steps of the analysis is included in each jupyter notebook. Notebooks starting with `UH` refer to cases denoted as Unkown Homeland. `ALL` is used for all other cases.

## Running the notebooks:

Before running the notebooks please `cd` into the `requirements` then

> `pip install - r requirements_(respective name).txt`

This will install the appropriate dependencies for each notebook. 

Please run first:

> `pip install - r requirements_utilities.txt`

This will install all dependencies used in the helpers functions found [in this subdirectory](https://github.com/jethronap/AsylumData_KU/tree/main/utilities).

### Disclaimer:

This project is still a work in progress.