import argparse
import csv
import cv2
import itertools
import numpy as np
import yaml
import os

def draw_tracks(img, x, y, dx, dy, vector_scale=60, circle_size=2, circle_color="yellow", line_width=2, line_color="red"):
    colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
            'yellow': [0, 255, 255], 'white': [255, 255, 255]}
    # draw the tracks
    #cv2.line(mask, (x, y), (int(x + vector_scale*dx), int(y + vector_scale*dy)), colormap[line_color], line)
    cv2.line(img, (x, y), (int(x + vector_scale*dx), int(y + vector_scale*dy)), colormap[line_color], line_width)
    #cv2.circle(mask, (x, y), point_size, colormap[circle_color], -1)
    cv2.circle(img, (int(x), int(y)), circle_size, colormap[circle_color], -1)


def lucas_kanade(file1, file2, output_path,
    vector_scale=60, circle_size=2, circle_color="yellow", line_width=2, line_color="red",
    save = True):

    print("compare ", file1, file2)

    conf_path = os.path.dirname(os.path.abspath(__file__)) + "/config.yaml"
    if not os.path.exists(output_path+"/csv/"):
        os.makedirs(output_path+"/csv/")

    config = yaml.load(open(conf_path), Loader=yaml.FullLoader)
    conf = config['LucasKanade']
    # colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
    #        'yellow': [0, 255, 255], 'white': [255, 255, 255]}
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = conf['quality_level'],
                          minDistance = 7,
                          blockSize = 7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize = (conf['window_size'],
                                conf['window_size']),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    img1 = cv2.imread(file1)
    img2 = cv2.imread(file2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)
    mask = np.zeros_like(img1)
    p1 = p0
    if p0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        data = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            dx = a - c
            dy = b - d
            # cv2.line(mask, (c, d), (int(c + vector_scale*dx), int(d + vector_scale*dy)), colormap[line_color], line)
            # cv2.line(img2, (c, d), (int(c + vector_scale*dx), int(d + vector_scale*dy)), colormap[line_color], line)
            # cv2.circle(mask, (c, d), point_size, colormap[circle_color], -1)
            # cv2.circle(img2, (c, d), point_size, colormap[circle_color], -1)
            if save:
                draw_tracks(img2, c, d, dx, dy, vector_scale, circle_size, circle_color, line_width, line_color)
            data.append([c, d, dx, dy])

    if save:
        filename = file1.split("/")
        filename = filename[len(filename)-1]
        temp = filename.split(".")

        output_file = output_path + "/" + temp[0] + ".png"
        print("saving", output_file)
        # cv2.imwrite(output_file, mask)
        cv2.imwrite(output_file, img2)    
        output_file = output_path + "/csv/" + temp[0] +".csv"
        with open(output_file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(data)

    results = {"image":img2, "vectors":data}
    return results

def farneback(file1, file2, vector_scale=1.0):
    colormap = {'blue': [255, 0, 0], 'green': [0, 255, 0], 'red': [0, 0, 255],
    'yellow': [0, 255, 255], 'white': [255, 255, 255]}
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    conf = config['Farneback']

    frame1 = cv2.imread(file1)
    prv = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame1)
    frame2 = cv2.imread(file2)
    nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prv, nxt, None, 0.5, 3,
                                        conf['window_size'],
                                        3, 5, 1.1, 0)
    height, width = prv.shape

    data = []
    for x, y in itertools.product(range(0, width, conf['stride']),
                                  range(0, height, conf['stride'])):
        if np.linalg.norm(flow[y, x]) >= conf['min_vec']:
            dy, dx = flow[y, x].astype(int)
            dx = args.vector_scale * dx
            dy = args.vector_scale * dy
            cv2.line(mask, (x, y), (x + int(dx), y + int(dy)), colormap[args.line_color], args.line)
            cv2.line(frame2, (x, y), (x + int(dx), y + int(dy)), colormap[args.line_color], args.line)
            cv2.circle(mask, (x, y), args.size, colormap[args.circle_color], -1)
            cv2.circle(frame2, (x, y), args.size, colormap[args.circle_color], -1)
            data.append([x, y, dx, dy])
    cv2.imwrite('vectors.png', mask)
    cv2.imwrite('result.png', frame2)
    with open('data.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
