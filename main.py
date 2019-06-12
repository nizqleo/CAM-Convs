import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np

import resn_CAM as RC
from data_reader import Sematics2D3D

# experiment parameters
numStage1=10
numStage2=50
numstage3=200
midpoint= 25


h_size=96
w_size=96
f_rate=1
usecam=1

# data loading/prepareing/augmentation
transform = transforms.Compose([
    transforms.CenterCrop((f_rate*h_size,f_rate*w_size)),
    transforms.Resize((h_size,w_size)),
    transforms.ToTensor()])

#'/media/nizq/My Passport/2D3Dsemantic'

trainset = Sematics2D3D('/media/nizq/My Passport/2D3Dsemantic', [1], transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, drop_last=False)

# model
if(usecam==1):
    model=RC.model_generator(h_size, w_size,f_rate)
else:
    model=RC.model_generator(h_size, w_size,f_rate,False)

# print(model)
# use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#           0      1     2     3     4  
# output = LR_1, MR_1, MR_2, HR_1, HR_2
def Loss(output, depth, norm, level, Use_gradient=False):
    depth_inv=1/depth
    total_Loss = 0
    
    dep_pr=output[level][:,0:1,:]
    con_pr=output[level][:,1:2,:]
    if(level <= 2):
        norm_pr=output[level][:,2:5,:]


    dep_pr=F.upsample(dep_pr, list(depth_inv.size()[-2:]), mode='bilinear', align_corners=True)
    con_pr=F.upsample(dep_pr, list(depth_inv.size()[-2:]), mode='bilinear', align_corners=True)
    if(level <= 2):
        norm_pr=F.upsample(norm, list(depth_inv.size()[-2:]), mode='bilinear', align_corners=True)
    
    Ld=torch.norm(dep_pr - depth_inv, p=1)
    Lg=0
    Lc=0
    Ln=0

    for p in range(16):
        for i in range(h_size):
            for j in range(w_size):
                
                # Loss of Confidence
                Lc+=abs(con_pr[p,:, i, j]-math.exp(-abs(dep_pr[p,:, i, j]-depth_inv[p,:, i, j])))

                # Loss of gradient
                if(Use_gradient):
                    for k in [1,2,4,8,16]:
                        dg=torch.zeros(2)
                        if i+k<h_size:
                            dg[0]=(dep_pr[p,:, i+k, j]-dep_pr[p,:, i, j])/abs(dep_pr[p,:, i+k, j]+dep_pr[p,:, i, j])-(depth_inv[p,:, i+k, j]-depth_inv[p,:, i, j])/abs(depth_inv[p,:, i+k, j]+depth_inv[p,:, i, j])

                        if j+k<w_size:
                            dg[1]=(dep_pr[p,:, i, j+k]-dep_pr[p,:, i, j])/abs(dep_pr[p,:, i, j+k]+dep_pr[p,:, i, j])-(depth_inv[p,:, i, j+k]-depth_inv[p,:, i, j])/abs(depth_inv[p,:, i, j+k]+depth_inv[p,:, i, j])

                        Lg+=torch.norm(dg, p=2)
                
                # Loss of norm
                if level <=2:
                    dif=torch.FloatTensor( [norm_pr[p,0, i, j]-norm[p,0, i, j], norm_pr[p,1,i, j]-norm[p,1,i, j],norm_pr[p,2,i, j]-norm[p,2,i, j]] )
                    Ln+=torch.norm(dif, p=2)
    
    return 150*Ld+100*Lg+50*Lc+25*Ln
        
# initializa optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)

# stage 1
print("start stage 1 training...")
for epoch in range(numStage1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        print(i)
        # get the inputs; data is a list of [inputs, labels]
        rgbs, depth, norm= data[0].to(device), data[1].to(device), data[2].to(device)

        #rgbs, depth, norm= data[0], data[1], data[2]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(rgbs)
        loss = Loss(outputs, depth, norm, 0)
        scheduler.step(loss)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
            # print every 2000 mini-batches
        print(' loss: %.3f' %(running_loss))
        running_loss = 0.0

torch.save(model.state_dict(), "./stage_1.pt")
print("Stage 1 training done!")

optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.99))
scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)
# stage 2
print("start stage 2 training...")
for epoch in range(numStage2):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        rgbs, depth, norm= data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()
        outputs = model(rgbs)
        
        loss = Loss(outputs, depth, norm, 0)/3+Loss(outputs, depth, norm, 1)/2
        if(epoch > midpoint):
            loss += Loss(outputs, depth, norm, 1, True)
        else:
            loss += Loss(outputs, depth, norm, 1)
        scheduler.step(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        print(' loss: %.3f' %(running_loss))
        running_loss = 0.0

torch.save(model.state_dict(), "./stage_2.pt")
print("Stage 2 training done!")


optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)
# stage 3
print("start stage 3 training...")
for epoch in range(numStage3):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        rgbs, depth, norm= data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()
        outputs = model(rgbs)
        
        loss = Loss(outputs, depth, norm, 0)/5+ Loss(outputs, depth, norm, 1)/4
        if(epoch > midpoint):
            loss += Loss(outputs, depth, norm, 2, True)/3+Loss(outputs, depth, norm, 3, True)/2+Loss(outputs, depth, norm, 4, True)
        else:
            loss += Loss(outputs, depth, norm, 2)/3+Loss(outputs, depth, norm, 3)/2+Loss(outputs, depth, norm, 4)
        scheduler.step(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #if i % 2000 == 1999:    # print every 2000 mini-batches
        print(' loss: %.3f' %(running_loss))
        running_loss = 0.0

torch.save(model.state_dict(), "./stage_3.pt")
print("Stage 3 training done!")


### validation

h_size=128
w_size=128
f_rate=1
usecam=1

transform2 = transforms.Compose([
    transforms.CenterCrop((f_rate*h_size,f_rate*w_size)),
    transforms.Resize((h_size,w_size)),
    transforms.ToTensor()])


testset = Sematics2D3D('/media/nizq/My Passport/2D3Dsemantic', [1], transforms=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2, drop_last=False)

l=0

# model
if(usecam==1):
    model_test=RC.model_generator(h_size, w_size,f_rate)
else:
    model_test=RC.model_generator(h_size, w_size,f_rate,False)

model_test.load_state_dict(torch.load("./stage_3.pt"), strict=False)

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        rgbs, depth, norm = data[0].to(device), data[1].to(device), data[2].to(device)
        output = model_test(rgbs)
        depth_inv=1/depth
        
        l+=torch.norm(depth_inv-output[level][0], p=1)
        

print('depth l1 inv error: %d %%' % (
    l/testset.__len__()))
