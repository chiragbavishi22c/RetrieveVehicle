import os
import cv2
import argparse
import json

def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def main():
    bbox_color = (255, 0, 0)

    test_tracks_root = os.path.join('/home/satudent/7. seminar/bounding-test-tracks.json')
    with open(test_tracks_root, "r") as f:
        test_tracks = json.load(f)
    
    all_tracks_root = os.path.join('/home/satudent/7. seminar/data/train-tracks.json')
    with open(all_tracks_root, "r") as f:
        all_tracks = json.load(f)

    test_keys = list(test_tracks.keys())

    for i in test_keys:
        listFrames = all_tracks[str(i)]['frames']
        listBoxes = all_tracks[str(i)]['boxes']

    vPath = '/home/satudent/7. seminar' + listFrames[0][1:-16] + '/vdo.avi'
    i = 0
    frameNumber = int(listFrames[i][27:-4])
    total = len(listFrames)
     
    vidcap = cv2.VideoCapture(vPath) 
    success, image = vidcap.read()
    count = 1

    output_path = "/home/satudent/7. seminar/data/output123.mp4"
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while success:
        if(count == frameNumber):
            x, y , w, h = listBoxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), bbox_color, thickness=2)
            i = i + 1
            if i<total : 
                frameNumber = int(listFrames[i][27:-4])
                # cv2.imshow('frame', image)

            
        count = count + 1

        out.write(image)

        success, image = vidcap.read()

    # ttemp = all_tracks[test_tracks[0]]
    # video_root = os.path.join('/home/satudent/7. seminar/bounding-test-tracks.json')
    vidcap.release()
    out.release()

if __name__ == '__main__':
    print("Loading parameters...")

    main()
