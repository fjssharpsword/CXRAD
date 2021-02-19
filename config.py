import os
import torchvision.transforms as transforms

#config for common
config = {
            'CKPT_PATH': '/data/pycode/CXRAD/model/',
            'log_path':  '/data/pycode/CXRAD/log/',
            'img_path': '/data/pycode/CXRAD/imgs/',
            'CUDA_VISIBLE_DEVICES': "0,1,2,3,4,5,6,7",
            'MAX_EPOCHS': 20, 
            'BATCH_SIZE': 512,#256
            'TRAN_SIZE': 256,
            'TRAN_CROP': 224,
            'PROB': 0.5
         } 

transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_train = transforms.Compose([
   transforms.Resize((256,256)),
   #transforms.CenterCrop(224),
   transforms.RandomCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

#config for NIH-CXR dataset
PATH_TO_IMAGES_DIR_NIH = '/data/fjsdata/NIH-CXR/images/images/'
PATH_TO_BOX_FILE_NIH = '/data/pycode/CXRAD/dataset/fjs_BBox.csv'
PATH_TO_TRAIN_VAL_BENCHMARK_FILE_NIH = '/data/pycode/CXRAD/dataset/bm_train_val.csv'
PATH_TO_TEST_BENCHMARK_FILE_NIH = '/data/pycode/CXRAD/dataset/bm_test.csv'
CLASS_NAMES_NIH = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding']
#config for CVTE-CXR dataset
PATH_TO_IMAGES_DIR_CVTE = '/data/fjsdata/CVTEDR/images'
PATH_TO_TEST_FILE_CVTE = '/data/pycode/CXRAD/dataset/cvte_test.txt'
#config for Vin-CXR dataset
PATH_TO_IMAGES_DIR_Vin_Train = '/data/fjsdata/Vin-CXR/train_val_jpg/'
PATH_TO_IMAGES_DIR_Vin_Test = '/data/fjsdata/Vin-CXR/test_jpg/'
PATH_TO_TEST_FILE_VIN =  '/data/pycode/CXRAD/dataset/VinCXR_test.txt'
PATH_TO_TRAIN_FILE_VIN =  '/data/pycode/CXRAD/dataset/VinCXR_train.txt'
PATH_TO_VAL_FILE_VIN =  '/data/pycode/CXRAD/dataset/VinCXR_val.txt'
CLASS_NAMES_Vin = ['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No finding']


