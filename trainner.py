import cv2, os
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def getImagesAndLabels(path):
    # liste des chemins des images
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # création d'une liste de visage
    faceSamples = []
    # Liste des noms
    Ids = []
    # boucle pour charger les noms
    for imagePath in imagePaths:
        # conversion vers l'espace de niveau de gris
        pilImage = Image.open(imagePath).convert('L')
        # convertir la liste en une liste numpy(vecteur d'images)
        imageNp = np.array(pilImage, 'uint8')
        # obtenir le nom de la personne sur chaque image
        Id = int(os.path.split(imagePath)[-1].split(".")[0])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp,1.15,5)
        for (x, y, w, h) in faces:
            faceSamples.append(cv2.resize(imageNp[y:y + h, x:x + w],(400,400)))
            Ids.append(Id)
            cv2.imshow("trainning", cv2.resize(imageNp[y:y + h, x:x + w],(400,400)))
            cv2.waitKey(1)
    return faceSamples, Ids

faces, Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.write('trainner/trainner.yml')
cv2.destroyAllWindows()
