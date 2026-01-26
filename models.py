from ultralytics import RTDETR

def load_model():
    model = RTDETR("rtdetr-x.pt")
    return model