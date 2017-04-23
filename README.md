# ManateeSketchIdentification
This is an academic research project done for the course EEL 6935 Big Data Ecosystems. The goal behind this project is to identify manatees based on their scar pattern.

The steps should be carried out in the order of the following files.
-> data_augmentation.ipynb
--> The seven kinds of augmentations are created using this file and are stored in the specified folders which need to be created.
   
-> resize_files.ipynb
--> The code is used to resize the images to sizes of 256x128 and 128x64 and the resized images are stored in the specified folders which need to be created.
   
-> create_dataset.ipynb
--> The code is used to store the resized images and labels in files [train_mini_final.txt, train_labels_mini_final.txt, test_mini_final.txt, test_labels_mini_final.txt, train_final.txt, train_labels_final.txt, test_final.txt, test_labels_final.txt].
   
-> 3_conv_layers.ipynb
--> The code is used to read the files [train_mini_final.txt, train_labels_mini_final.txt, test_mini_final.txt, test_labels_mini_final.txt] created in the previous step.
--> One hot labels are created for the train and test labels.
--> The layers for the 8 layer network are defined.
--> The training takes place for 200 epochs. 
--> The batch size is 203. 
--> The test set acts as the validation set and the accuracy for the test set is computed after every epoch (for top1, top5, top10 and top20). 
--> The model is saved after every epoch to model-3.ckpt and can be retrieved from it (in case of failure).
--> The accuracies for top1, top5, top10 and top20 are stored in top_1_list, top_5_list, top_10_list and top_20_list and can be retrieved from them.
--> The results were printed in 3_conv_layers.ipynb to get an idea about how the training and test accuracies were being computed.
