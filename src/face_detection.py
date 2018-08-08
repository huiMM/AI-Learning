'''
@author: xusheng
'''

import cv2
import os
from utils import SequenceGenerator


class CameraReader(object):

    def __init__(self):
        # TODO init neuro model
        self._g_sequence = SequenceGenerator(1)
    
    def _detect_eyes(self, roi_color, roi_gray, eye_cascade):
        # detect and draw border of eyes with green rectangle
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.03, minNeighbors=5, flags=0, minSize=(35, 35))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return len(eyes) > 0
    
    def _detect_face(self, frame_color, frame_gray, face_cascade, eye_cascade=None):
        # detect and draw border of face with blue rectangle
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = frame_color[y:y+h, x:x+w]
            roi_gray = frame_gray[y:y+h, x:x+w]
            
#             if self._detect_eyes(roi_color, roi_gray, eye_cascade):
#                 pass
            
            # TODO model.predict(roi_gray), if prob > 0.7 then load name
            label_text = 'Face Detected'
            # PSEUDO CODE
            # prob = self._model.predict(roi_gray)
            # if prob > 0.7:
            #     label_text = LABEL_NAME
            # else:
            #     label_text = 'UNKNOWN'
            
            cv2.putText(img=frame_color, text=label_text, org=(x, y-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(127, 127, 127), thickness=1)

#             self._save_as('full_%d.jpg' % self._g_sequence.next(), frame_color, size=None)
#             self._save_as('128x128_2_%d.jpg' % self._g_sequence.next(), roi_color)
        
        return len(faces) > 0

    def build_camera(self):
        face_cascade = cv2.CascadeClassifier(os.path.join('..', 'data', 'opencv', 'cascades', 'haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(os.path.join('..', 'data', 'opencv', 'cascades', 'haarcascade_eye.xml'))
    
        camera = cv2.VideoCapture(0)
        size = (camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        success, frame_color = camera.read()
        while success:
            success, frame_color = camera.read()
            frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
            
            if self._detect_face(frame_color, frame_gray, face_cascade, eye_cascade):
                pass
    #             cv2.imwrite(os.path.join('..', 'data', 'opencv', 'saved', 'demo.jpg'), frame_color)
            
            cv2.imshow(("camera_%sx%s" % (size[0], size[1])), frame_color)
        
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
        
        camera.release()
        cv2.destroyAllWindows()
    
    def _save_as(self, name, roi, size=(128, 128)):
        f = roi
        if size is not None:
            f = cv2.resize(roi, size)
        cv2.imwrite(os.path.join('..', 'data', 'opencv', 'saved', name), f)


if __name__ == '__main__':
    camera = CameraReader()
    camera.build_camera()