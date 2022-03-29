import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
# Naloži slike iz mape images
sfr.load_encoding_images("images/")

# Vklopi kamero računalnika
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Pridobivanje koodrinat obraza
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Izpis kvadrata in imena na obrazu
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 201), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()