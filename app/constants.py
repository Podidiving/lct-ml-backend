MODEL_IMGSZ = 256
DETECTOR_IMGSZ = 864
MODEL_WEIGHTS = "/app/app/weights/classify.pt"
DETECTOR_WEIGHTS = "/app/app/weights/fine.pt"
INDEX_TO_LABEL = {
    0: "Acer",
    1: "Alnus",
    2: "Betula",
    3: "Caragana",
    4: "Corylus",
    5: "Euonymus",
    6: "Frangula",
    7: "Fraxinus",
    8: "Lonicera",
    9: "Picea",
    10: "Pinus",
    11: "Populus",
    12: "Prunus",
    13: "Quercus",
    14: "Reynoutria",
    15: "Rosa",
    16: "Salix",
    17: "Sambucus",
    18: "Sorbus",
    19: "Tilia",
    20: "Ulmus",
    21: "Viburnum",
}
DETECTOR_INDEX_TO_LABEL = {
    0: "Crack",
    1: "Duplo",
    2: "Treschina",
    3: "Plodovoe_telo",
    4: "Gnil",
}
