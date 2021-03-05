# Diversity-Algorithm-CLI-Tool
CLI tool to Diversity Algorithm (Noob Friendly)


Simply run the ```da-picker.py``` to index your data with the following arguments:
* --INPUT_FILE_PATH : Path to Pkl or CSV File
* --UMAP : Enable UMAP for Diversity Embeddings
* --subset_size : Percentage of points in the entire dataset to selected by the Diversity Algorithm
* --OUTPUT_FILE_PATH : File path for the Diversity Algorithm Output. Generates a '.pkl' file if extension is mentioned in the argument, else creates a new dataset 
* --metric : Prints out a Class Distribution

and a few other optional arguments (check file).

Example: 
```bash
!python /content/SpaceForceDataSearch/da-picker.py --UMAP True --subset_size 0.5 --INPUT_FILE_PATH "/content/embedding_path_tuple.pkl" --OUTPUT_FILE_PATH "/content/embedding_path_output" --metric True
```

## Dependencies

Install the necessary packages from requirements.txt using ```pip install -r requirements.txt``` before you run the scripts.

## TODO
- [ ] Add Pt and Tf Model Support 
- [ ] Add DA Variant   
- [ ] Bring in Packag 




