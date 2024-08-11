'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch
import random
import torchvision
import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
#count =0

def detect_faces(img: torch.Tensor,img_path:str) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    img_new = img.permute(1,2,0)
    img_new_np=img_new.cpu().numpy()
    face_location = face_recognition.face_locations(img=img_new_np,number_of_times_to_upsample=0,model="cnn")
    det_res = []
    for i in range (0,len(face_location)):
        if (len(face_location)==0):
            print ("No face detected")
        else:
            top, right, bottom, left = face_location[i]
            face_box = [left,top, right-left, bottom-top]
            face_box_fl = [float(x) for x in face_box]
            det_res.append(face_box_fl)
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.
    detection_results = det_res
    # x,y,w,h = det_res[1]
    # bounding = torch.tensor([[x,y,x+w,y+h]])
    # img_new = torchvision.utils.draw_bounding_boxes(img, bounding,colors=(255,0,0))
    # print (img_path)
    # print (detection_results)
    # print (img.shape)
    # show_image(img_new)
    # TODO: Add your code here. Do not modify the return and input arguments.
    
    return detection_results


def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.
    encoding={}
    for i in imgs.keys():
        img_new=imgs[i].permute(1,2,0).cpu().numpy()
        feature = face_recognition.face_encodings(img_new)
        #print (len(feature[0]))
        encoding[i]=feature[0]
    #print (encoding)
    # TODO: Add your code here. Do not modify the return and input arguments.
    K=5
    center_img=[]
    ran_num = torch.randint(1,36,(K,))
    print (ran_num)
    img_name = list(encoding.keys())
    for kuch_i in range(K):
        center_img.append(img_name[ran_num[kuch_i]])
    #center_img = random.sample(encoding.keys(),K)
    print (center_img)
    center=[]
    for kuch_toh in center_img:
        center.append(encoding[kuch_toh])

    iteration=0
    stop_clustering=False
    while (not stop_clustering):
        print ("~~~~~~~~!!!!@@@@@@Iteration"+str(iteration)+"@@@@@@!!!!~~~~~~~~")
        iteration+=1
        centroid_enc_map = {}
        cluster = {}
        for i in range(K):
            cluster[i]=[]
            centroid_enc_map[i]=torch.tensor(center[i])
        for dp in encoding.keys():
            dist={}
            for c in centroid_enc_map.keys():
                #sqrt ( x1-x2**2)+(y1-y2**2)
                if (dp in center_img):
                    print (dp+"found in list")
                    euc_dist = torch.sqrt(torch.sum(torch.square(torch.tensor(encoding[dp])-torch.tensor(centroid_enc_map[c]))))
                    print (euc_dist)
                euc_dist = torch.sqrt(torch.sum(torch.square(torch.tensor(encoding[dp])-torch.tensor(centroid_enc_map[c]))))
                ## I need the mapping from the centroid_encoding to the cluster center.
                dist[c]=euc_dist
            #print (dist)
            min_euc_dist = min(list(dist.values()))
            #closest_centroid_encoding=list(filter(lambda x:dist[x]==min_euc_dist,dist))[0]
            #print ("Printing Cluster")
            #print (cluster)
            for key in dist.keys():
                if (dist[key]==min_euc_dist):
                    closest_centroid_encoding = key
            cluster[closest_centroid_encoding].append(dp)
        
        ### Recaluculate the Centroid
        prev_centroid = center
        print (cluster)
        center = []
        for centroid in cluster.keys():
            noofdpincluster = len(cluster[centroid])
            if noofdpincluster==0:
                continue
            sum_arr = [0 for i in range(128)]
            for cluster_dp in cluster[centroid]:
                sum_arr=sum_arr+encoding[cluster_dp]
            new_cluster_center = [x / noofdpincluster for x in sum_arr]
            center.append(new_cluster_center)
        
        print (len(center))
        print (len(prev_centroid))
        ## Convergence break wala logic - 
        for i in range(len(center)):
            stop_clustering=True
            el_center = torch.tensor(center[i])
            el_pc = torch.tensor(prev_centroid[i])
            if (torch.all(el_center.eq(el_pc))):
                continue
            else:
                    stop_clustering=False
    print ("FINAL ANSWER...LOCK KIya Jaaye:")
    print (cluster)
    
    cluster_results=list(cluster.values())
    #while not stop:
           # Perform K means algorithm

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)