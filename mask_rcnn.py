import cv2
import numpy as np
import random
import torch
import torchvision
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms


# Using COCO dataset with 91 categories and >200K labeled images
# label names are in a text file
label_file = open('labels.txt', 'r')
lines = label_file.readlines()
categories=[]
for line in lines:
    categories.append(str(line.strip()))

# Assign each category a random color
colors = np.random.uniform(0, 255, size=(len(categories), 3))

def mask_rcnn(image):
    #print(len(categories))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # From pytorch.org : If True, returns a model pre-trained on COCO train2017
    maskrcnn_resnet50_fpn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
    maskrcnn_resnet50_fpn.to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Transform user's input image
    image_trans = transform(image)
    # Put image on device
    image_trans = image_trans.unsqueeze(0).to(device)
    # Feed image to model and get outputs
    out = maskrcnn_resnet50_fpn(image_trans)

    # Get model scores for the user's input image
    scores = []
    for score in out[0]['scores']:
        scores.append(score)

    # Check which scores are above a certain confidence level (80% for now)
    # I kept the level low since users might want more objects to move even if some are less accurate
    # Will only display these masks/boxes to the user
    predictions = 0
    for score in scores:
        if score > .8:
            predictions+=1

    #print(out[0]['masks'].size())
    #exit()
    # The masks are size [# masks, 1, image_height, image_width]
    # Squeeze will remove the empty dimension to make it [# masks, image_height, image_width]
    # detach and cpu are needed to get the tensor data from the gpu
    output_masks = out[0]['masks'].squeeze().detach().cpu().numpy()
    masks=np.zeros_like(output_masks)
    for i in range(np.shape(masks)[0]):
        for j in range(np.shape(masks)[1]):
            for k in range(np.shape(masks)[2]):
                # Perform thresholding so the mask is seperated by background/foreground to black/white
                if output_masks[i][j][k] > .5:
                    masks[i][j][k] = 1
    # Take the x highest score masks, where x = #predictions
    masks = masks[:predictions]
    print(np.shape(masks))
    #exit()
    boxes=[]
    # detach and cpu are needed to get the tensor data from the gpu
    output_boxes = out[0]['boxes'].detach().cpu()
    for box in output_boxes:
        boxes.append([(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))])
    # Take the x highest score boxes, where x = #predictions
    boxes = boxes[:predictions]

    labels = []
    output_labels = out[0]['labels']
    # Convert label number to label name (list text file provided from coco dataset github)
    for label in output_labels:
        labels.append(categories[label])

    # Add each mask/box/label onto the image
    for i in range(len(masks)):
        r = np.zeros_like(masks[i]).astype(np.uint8)
        g = np.zeros_like(masks[i]).astype(np.uint8)
        b = np.zeros_like(masks[i]).astype(np.uint8)
        # Need it later for cv2, since generating random, can't generate again
        mask_color = colors[random.randrange(0, len(colors))]
        r[masks[i] == 1], g[masks[i] == 1], b[masks[i] == 1]  = mask_color
        colored_mask = np.stack([r, g, b], axis=2)
        image = np.array(image)
        # Add mask
        # blend the image and segmentation map
        # cv2.addWeighted(source1, alpha, source2, beta, gamma[, dst[, dtype]])
        # alpha weight of first image
        # beta is weight of second image, I've kept it fairly low so the masks aren't very bright
        # gamma is scalar addition, but I am not adding any
        cv2.addWeighted(image, 1, colored_mask, .25, 0, image)
        # Add box
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=mask_color, thickness=1)
        text = labels[i]
        bottom_left_corner_location = (boxes[i][0][0], boxes[i][0][1])
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .4
        thickness = 1
        color = (0,0,0)
        # Add text
        # (img, text, bottom left corner of text, font, font scale, color, thickness, line type)
        cv2.putText(image, text, bottom_left_corner_location, fontFace, fontScale, color, thickness, cv2.LINE_AA)
    return image, masks, boxes, labels




