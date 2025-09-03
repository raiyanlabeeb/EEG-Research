import torch
import torch.nn as nn
import os
from datetime import datetime
from eegModelUtil import preprocess_eeg_data, get_cv_split, get_device, load_data, train_model, evaluate_model, get_parameters, plot_training_curves, output_confusion_matrix
class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)
            
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)
 
class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        # if self.bias:
        #     self.bias.data.fill_(0.0)
 
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
 
    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class TemporalInception(nn.Module):
    def __init__(
        self,
        in_chan=1,
        kerSize_1=(1, 3),
        kerSize_2=(1, 5),
        kerSize_3=(1, 7),
        kerStr=1,
        out_chan=4,
        pool_ker=(1, 3),
        pool_str=1,
        bias=False,
        max_norm=1.0,
    ):

        super(TemporalInception, self).__init__()

        self.conv1 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_1,
            stride=kerStr,
            padding="same",
            groups=out_chan,
            bias=bias,
            max_norm=max_norm,  # type: ignore
        )

        self.conv2 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_2,
            stride=kerStr,
            padding="same",
            groups=out_chan,
            bias=bias,
            max_norm=max_norm,  # type: ignore
        )

        self.conv3 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kerSize_3,
            stride=kerStr,
            padding="same",
            groups=out_chan,
            bias=bias,
            max_norm=max_norm,  # type: ignore
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size=pool_ker,
            stride=pool_str,
            padding=(
                round(pool_ker[0] / 2 + 0.1) - 1,
                round(pool_ker[1] / 2 + 0.1) - 1,
            ),
        )
        self.conv4 = Conv2dWithConstraint(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            groups=out_chan,
            bias=bias,
            max_norm=max_norm,  # type: ignore
        )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        out = torch.cat((p1, p2, p3, p4), dim=1)  # type: ignore
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
 
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
 
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalBlock, self).__init__()
        self.conv1 = Conv1dWithConstraint(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                          dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
 
        self.conv2 = Conv1dWithConstraint(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, 
                                          dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(num_features=n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
 
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()
 
    #     self.init_weights()
 
    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)
 
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = out+res
        out = self.relu(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout, bias=bias, WeightNorm=WeightNorm, max_norm=max_norm)]
 
        self.network = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.network(x)

class TCNet(nn.Module):
    """ ""
    Input shape:
    Output shape:
    """

    def __init__(
        self,
        F1=32,
        D=2,
        kerSize=32,
        eeg_chans=22,
        poolSize=8,
        kerSize_Tem=4,
        dropout_dep=0.5,
        dropout_temp=0.5,
        dropout_atten=0.3,
        tcn_filters=64,
        tcn_kernelSize=4,
        tcn_dropout=0.3,
        n_classes=4,
    ):
        super(TCNet, self).__init__()
        self.F2 = F1 * D

        self.sincConv = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kerSize),
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn_sinc = nn.BatchNorm2d(num_features=F1)

        self.conv_depth = Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(eeg_chans, 1),
            groups=F1,
            bias=False,
            max_norm=1.0,  # type: ignore
        )
        self.bn_depth = nn.BatchNorm2d(num_features=self.F2)
        self.act_depth = nn.ELU()
        self.avgpool_depth = nn.AvgPool2d(
            kernel_size=(1, poolSize), stride=(1, poolSize)
        )
        self.drop_depth = nn.Dropout(p=dropout_dep)

        self.incept_temp = TemporalInception(
            in_chan=self.F2,
            kerSize_1=(1, kerSize_Tem * 4),
            kerSize_2=(1, kerSize_Tem * 2),
            kerSize_3=(1, kerSize_Tem),
            kerStr=1,
            out_chan=self.F2 // 4,
            pool_ker=(3, 3),
            pool_str=1,
            bias=False,
            max_norm=0.5,
        )
        self.bn_temp = nn.BatchNorm2d(num_features=self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d(
            kernel_size=(1, poolSize), stride=(1, poolSize)
        )
        self.drop_temp = nn.Dropout(p=dropout_temp)

        self.tcn_block = TemporalConvNet(
            num_inputs=self.F2,
            num_channels=[tcn_filters, tcn_filters],  # [64,64] 与滤波器数量一致
            kernel_size=tcn_kernelSize,  # 4
            dropout=tcn_dropout,
            bias=False,
            WeightNorm=True,
            max_norm=0.5,
        )

        self.flatten = nn.Flatten()
        self.liner_cla = LinearWithConstraint(
            in_features=tcn_filters,
            out_features=n_classes,
            max_norm=0.25,  # type: ignore
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Remove extra dimension if present
        if x.dim() == 4:
            x = x.squeeze(1)
        
        # Input x shape: (batch, channels, timepoints) - e.g., (64, 24, 1001)
        # Expected by model: (batch, timepoints, channels) - e.g., (64, 1001, 24)
        x = x.permute(0, 2, 1)  # (batch, timepoints, channels)
        
        # Add channel dimension for Conv2d operations
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)  # (batch, 1, timepoints, channels)
        
        x = self.sincConv(x)
        x = self.bn_sinc(x)

        x = self.conv_depth(x)  # (batch, F1*D, 1, channels), Conv2dWithConstraint, depthwise convolution from EEG-Net
        x = self.drop_depth(
            self.avgpool_depth(self.act_depth(self.bn_depth(x)))
        )  # (batch, F1*D, 1, 1),

        x = self.incept_temp(x)
        x = self.drop_temp(
            self.avgpool_temp(self.act_temp(self.bn_temp(x)))
        )  # (batch, F1*D, 1, 15)

        x = torch.squeeze(x, dim=2)  # (batch, F1*D, 15)
        # Also squeeze the last dimension to get (batch, channels, time)
        if x.dim() == 4:
            x = torch.squeeze(x, dim=-1)
        x = self.tcn_block(x)

        x = x[:, :, -1]
        x = self.flatten(x)
        x = self.liner_cla(x)  # (batch, n_classes)
        out = self.softmax(x)

        return out

def create_model(num_channels=24, num_classes=2):
    """Create EEGTCNet model optimized for your data"""
    return TCNet(
        F1=32,           # Number of temporal filters
        D=2,             # Depth multiplier
        kerSize=16,      # Reduced temporal kernel size (was 32)
        eeg_chans=num_channels,  # Your 24 channels
        poolSize=4,      # Reduced pooling size (was 8)
        kerSize_Tem=2,   # Reduced temporal inception kernel (was 4)
        dropout_dep=0.5, # Depthwise dropout
        dropout_temp=0.5, # Temporal dropout
        dropout_atten=0.3, # Attention dropout
        tcn_filters=64,  # TCN filters
        tcn_kernelSize=2, # Reduced TCN kernel size (was 4)
        tcn_dropout=0.3, # TCN dropout
        n_classes=num_classes  # Your 2 classes
    )

def start():
    """
    Main function with evaluation strategy from global variable.
    
    Args:
        test_subject (int): Subject index for LOSO (0-indexed)
    """
    
    NUM_SUBJECTS, NUM_TRIALS, NUM_CHANNELS, NUM_TIMEPOINTS, CLASSES, NUM_CLASSES, BATCH_SIZE, T_BATCH_SIZE, LR, NUM_EPOCHS, EVAL_TYPE, TEST_SUBJECT = get_parameters()
    file_name = os.path.basename(__file__)
    print(f'~~~MODEL: {file_name}~~~')
    print(f"~~~TESTING CLASSIFICATION FOR {CLASSES}~~~\n")
    print(f"~~~PARAMETERS: "
        f"\n  Number of Subjects: {NUM_SUBJECTS}"
        f"\n  Training Batch Size: {BATCH_SIZE}"
        f"\n  Testing Batch Size: {T_BATCH_SIZE}"
        f"\n  Learning Rate: {LR}"
        f"\n  Epochs: {NUM_EPOCHS}"
        f"\n  Evaluation Type: {EVAL_TYPE.upper()}"
        f"\n  Test Subject: {TEST_SUBJECT + 1 if EVAL_TYPE == 'leave_one_subject_out' else 'N/A'}\n~~~")
    
    eeg_data, eeg_labels = preprocess_eeg_data()
    print(f"PREPROCESSING COMPLETE. \nEEG data: {len(eeg_data)} samples, labels: {len(eeg_labels)}")
    print(f"Evaluation type: {EVAL_TYPE}")
    
    # Choose evaluation strategy based on global setting
    if EVAL_TYPE == "within_subject":
        train_data, test_data, train_labels, test_labels = get_cv_split(
            eeg_data, eeg_labels, n_splits=5, fold=2, evaluation_type="within_subject"
        )
    else:  # leave_one_subject_out
        train_data, test_data, train_labels, test_labels = get_cv_split(
            eeg_data, eeg_labels, fold=TEST_SUBJECT, evaluation_type="leave_one_subject_out"
        )
    
    print(f"TRAINING AND TESTING DATA SPLIT COMPLETE.")
    train_loader, val_loader, test_loader = load_data(train_data, train_labels, test_data, test_labels)
    print(f"LOADING DATA COMPLETE.")
    device = get_device()
    model = create_model(NUM_CHANNELS, NUM_CLASSES)
    print(f"MODEL CREATED.")
    print("TRAINING STARTED...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    print(f"TRAINING COMPLETE.")
    
    # Plot training curves
    print("PLOTTING TRAINING CURVES...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model/CodeEEGModels/training_curves/EEGTCNet_{timestamp}.png"
    
    plot_training_curves(
        train_losses, val_losses, 
        model_name="EEGTCNet",
        save_path=save_path,
        evaluation_type=EVAL_TYPE,
        classes=CLASSES,
        lr=LR,
        epochs=NUM_EPOCHS
    )
    
    preds, labels = evaluate_model(model, test_loader, device)
    print(f"EVALUATION COMPLETE.")
    
    output_confusion_matrix(preds, labels, CLASSES, file_name)
    

if __name__ == '__main__':
    start()