### UBC Adult Kidney Dataset

1. Request access (See [GitHub](https://github.com/rsingla92/kidneyUS/))
2. Download images from OneDrive folder, and rename folder to `raw` and store in `ViewLabeling/UBC_AdultKidneyDataset/raw/` (create subdir if needed)
3. Download label files `reviewed_labels_1.csv` and `reviewed_labels_2.csv` from [GitHub](https://github.com/rsingla92/kidneyUS/blob/main/labels/), and copy them to a `ViewLabeling/Datasheets/raw/ubc_adult`
4. Run the following scripts to preprocess the UBC adult kidney data
```
# a. Preprocess images
python -m src.data_prep.scripts.prep_ubc_kidney_dataset prep_images
# b. Preprocess metadata
python -m src.data_prep.scripts.prep_ubc_kidney_dataset prep_metadata
# c. Final touches on metadata
python -m src.data_prep.scripts.create_metadata clean_metadata
```