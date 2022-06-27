## Documentation

Steps

1. Download iNaturalist OpenData Metadata from https://github.com/inaturalist/inaturalist-open-data
    * instructions are in Metadata/Download]
    * use commands of the form `aws s3 cp s3://inaturalist-open-data/photos.csv.gz photos.csv.gz`
    * The photos.csv.gz file takes a while to download

2. Open metadata files

3. Get Eucalyptus taxon, and all child taxa. 

4. Join with observations table on taxon_id. 

5. Then, join with photo table on observation_uuid. Get photo_id and extension for each photo in a df

6. Download 1000 photo (by using `subprocess.run` and the `aws s3 cp` applied to the df from previous step). I just let it run and stopped after ~1000, might be better to randomly sample 1000 indices

7. Download list of all taxa in the GlobalTreeSearch database (DOI: 10.13140/RG.2.2.33593.90725)
    * https://tools.bgci.org/global_tree_search.php 
    * had to remove the two extra pieces of info at the top two rows (citation and doi) so that it was able to be processed by pandas

8. get taxon_id's for all 60k tree taxa

9. remove taxon_ids that are in the eucalyptus list from the tree list to get the other tree category

10. 