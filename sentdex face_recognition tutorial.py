import face_recognition
import cv2
import  os


KNOWN_FACES_DIR = 'Dataset/known_faces'
UKNOWN_FACES_DIR = 'Dataset/unknown_people'
TOLERANCE = 1
FRAME_THICKNESS = 3
FONT_THOCKNESS = 2
MODEL = 'cnn'

print('loading known faces')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    # load every file of faces of known persons
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        # load an image
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        # get 128-dimensions face encoding
        # always return a list of found faces
        print(image[0])
        encoding = face_recognition.face_encodings(image[0])

        # append encodings
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces")
# loop over a folder of faces you want to label
for filename in os.listdir(UKNOWN_FACES_DIR):
    # load image
    print(f'Filename{filename}', end=' ')

    image = face_recognition.load_image_file(f'{UKNOWN_FACES_DIR}/{filename}')

    # first grab the face locations
    locations = face_recognition.face_locations(image, model=MODEL)
    # since we know locations we can pass them to face encodings as second argument
    encoding = face_recognition.face_encodings(image, locations)
    # convert to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f', found{len(encoding)} face(s)')

    for face_encoding, face_location in zip(encoding, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        print(results)
