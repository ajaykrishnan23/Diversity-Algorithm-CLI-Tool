# Diversity-Algorithm-CLI-Tool
CLI tool to Diversity Algorithm (Noob Friendly)


Simply run the ```da-picker.py``` to index your data with the following arguments:
* --INPUT_FILE_PATH : path to pkl/csv file in format [embedding, filename] or a .pt file
* --UMAP : Enable UMAP for Diversity Embeddings
* --subset_size : Percentage of points in the entire dataset to selected by the Diversity Algorithm
* --OUTPUT_FILE_PATH : File path for the Diversity Algorithm Output. Generates a '.pkl' file if extension is mentioned in the argument, else creates a new dataset if a folder path is given
* --metric : Prints out a Class Distribution
* --img_size : Size of Images (Required when passing model)
* --embedding_size: Size of model's output embedding (Required when passing model)

and a few other optional arguments (check file).

Example: 
```bash
!python /content/Diversity-Algorithm-CLI-Tool/da-picker.py --DATA_PATH /content/UCMerced_LandUse/Images --img_size 64 --embedding_size 2048 --UMAP True --subset_size 0.01  --INPUT_FILE_PATH "/content/pt_model.pt" --OUTPUT_FILE_PATH "/content/embedding_path_output" --metric True
```

## Dependencies

Install the necessary packages from requirements.txt using ```pip install -r requirements.txt``` before you run the scripts.

## TODO
- [ ] Add Pt Model Support 
- [ ] Tf Model Support 
- [ ] Add DA Variant   
- [ ] Bring in Package
- [ ] Multiprocessing
- [ ] Code Restructuring




