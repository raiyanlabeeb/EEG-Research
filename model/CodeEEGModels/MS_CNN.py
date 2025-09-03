import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from eegModelUtil import preprocess_eeg_data, get_cv_split, get_device, load_data, train_model, evaluate_model, get_parameters
import torch
import torch.nn as nn

class MSCNN_Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,14,(1,5),(1,1),padding=(0,2)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1,14,(1,3),(1,1),padding=(0,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1,14,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
            nn.MaxPool2d((1,15),(1,15)),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d((1,15),(1,15)),
            nn.Conv2d(1,14,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(),
        )

        self.cl = nn.Sequential(
            nn.Conv2d(56,112,(1,1),(1,1)),  #in_channel , out_channel , kernel_size , stride
            nn.ReLU(), 
            nn.MaxPool2d((1,15),(1,15)),
        )

        self.fc = nn.Sequential(
            nn.Linear(448*3*3,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,2)

        )
    
    def forward(self,img):
        '''
        image : 5d-array (trials x freband x channels x height x width)
        '''
        

        feature1 = self.conv1(img[:,0,:,:,:])
        feature2 = self.conv2(img[:,0,:,:,:])
        feature3 = self.conv3(img[:,0,:,:,:])
        feature4 = self.conv4(img[:,0,:,:,:])
        
        feature = torch.cat((feature1,feature2,feature3,feature4),dim=1)

        fre1_feat = self.cl(feature)


        feature1 = self.conv1(img[:,1,:,:,:])
        feature2 = self.conv2(img[:,1,:,:,:])
        feature3 = self.conv3(img[:,1,:,:,:])
        feature4 = self.conv4(img[:,1,:,:,:])
        
        feature = torch.cat((feature1,feature2,feature3,feature4),dim=1)

        fre2_feat = self.cl(feature)

        feature1 = self.conv1(img[:,2,:,:,:])
        feature2 = self.conv2(img[:,2,:,:,:])
        feature3 = self.conv3(img[:,2,:,:,:])
        feature4 = self.conv4(img[:,2,:,:,:])
        
        feature = torch.cat((feature1,feature2,feature3,feature4),dim=1)

        fre3_feat = self.cl(feature)

        feature1 = self.conv1(img[:,3,:,:,:])
        feature2 = self.conv2(img[:,3,:,:,:])
        feature3 = self.conv3(img[:,3,:,:,:])
        feature4 = self.conv4(img[:,3,:,:,:])
        
        feature = torch.cat((feature1,feature2,feature3,feature4),dim=1)

        fre4_feat = self.cl(feature)

        feature_conb = torch.cat((fre1_feat,fre2_feat,fre3_feat,fre4_feat),dim=1)
        
        '''
        print('img size : ',img.shape)
        print('freband 1 feature shape: ',fre1_feat.size())
        print('feature shape: ',feature_conb.size())
        '''
        output = self.fc(feature_conb.reshape(img.shape[0],-1))
        return output

def create_model():
    return MSCNN_Net()
def main():
    NUM_SUBJECTS, NUM_TRIALS, NUM_CHANNELS, NUM_TIMEPOINTS, CLASSES, NUM_CLASSES, BATCH_SIZE, T_BATCH_SIZE, LR, NUM_EPOCHS  = get_parameters()
    print(f"~~~TESTING CLASSIFICATION FOR {CLASSES}~~~\n")
    print(f"~~~PARAMETERS: "
        f"\n  Number of Subjects: {NUM_SUBJECTS}"
        f"\n  Training Batch Size: {BATCH_SIZE}"
        f"\n  Testing Batch Size: {T_BATCH_SIZE}"
        f"\n  Learning Rate: {LR}"
        f"\n  Epochs: {NUM_EPOCHS}\n~~~")
    eeg_data, eeg_labels = preprocess_eeg_data()
    print(f"PREPROCESSING COMPLETE. \nEEG data: {len(eeg_data)} samples, labels: {len(eeg_labels)}")
    train_data, test_data, train_labels, test_labels = get_cv_split(eeg_data, eeg_labels, n_splits=5, fold=2)
    print(f"TRAINING AND TESTING DATA SPLIT COMPLETE.")
    train_loader, val_loader, test_loader = load_data(train_data, train_labels, test_data, test_labels)
    print(f"LOADING DATA COMPLETE.")
    device = get_device()
    # model = create_model(NUM_CHANNELS, NUM_TIMEPOINTS, NUM_CLASSES, device)
    model = create_model()
    print(f"MODEL CREATED.")
    train_model(model, train_loader, val_loader, device)
    print(f"TRAINING COMPLETE.")
    evaluate_model(model, test_loader, device)
    print(f"EVALUATION COMPLETE.")

if __name__ == '__main__':
    main()
