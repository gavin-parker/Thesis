import numpy as np
import cv2
from tracker import Tracker
frame = None
trackers = []


def mouse_callback(event, x, y, flags, params):
    if event == 1:
        global frame
        global trackers
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.floodFill(frame, None, (x, y), (255, 255, 255), (3, 3, 3), (3, 3, 3), flags=cv2.FLOODFILL_MASK_ONLY)
        bbox = mask[3]
        tracker = Tracker(frame, bbox)
        trackers.append(tracker)


def show_video():
    cap = cv2.VideoCapture('test_video.mp4')
    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('rgb', mouse_callback)

    global frame
    global trackers
    while True:
        ret, frame = cap.read()
        update_trackers(frame, trackers)
        if ret:
            cv2.imshow('rgb', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def update_trackers(frame, trackers):
    for i, tracker in enumerate(trackers):
        if tracker is not None:
            tracker.update(frame)




if __name__ == "__main__":
    show_video()
