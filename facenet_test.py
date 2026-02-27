from keras_facenet import FaceNet

embedder = FaceNet()
embedder.model.save(r"E:\DA\FaceDetect\facenet_exported.h5")
print("Done")