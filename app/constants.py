MODEL_IMGSZ = 256
DETECTOR_IMGSZ = 864
MODEL_WEIGHTS = "/app/app/weights/classify.pt"
DETECTOR_WEIGHTS = "/app/app/weights/fine.pt"
INDEX_TO_LABEL = {
    0: "Клен ( Acer sp. )",
    1: "Ольха ( Alnus sp. )",
    2: "Береза ( Betula sp. )",
    3: "Желтая акация ( Caragana sp. )",
    4: "Лещина ( Corylus sp. )",
    5: "Бересклет ( Euonymus sp. )",
    6: "Крушина ( Frangula sp. )",
    7: "Ясень ( Fraxinus sp. )",
    8: "Жимолость ( Lonicera sp. )",
    9: "Ель ( Picea sp. )",
    10: "Сосна ( Pinus sp. )",
    11: "Тополь ( Populus sp. )",
    12: "Слива ( Prunus sp. )",
    13: "Дуб ( Quercus sp. )",
    14: "Рейнутрия ( Reynoutria sp. )",
    15: "Роза (шиповник) ( Rosa sp. )",
    16: "Ива ( Salix sp. )",
    17: "Бузина ( Sambucus sp. )",
    18: "Рябина ( Sorbus sp. )",
    19: "Липа ( Tilia sp. )",
    20: "Вяз ( Ulmus sp. )",
    21: "Калина ( Viburnum sp. )",
}
DETECTOR_INDEX_TO_LABEL = {
    0: "Трещина",
    1: "Дупло",
    2: "Гниль",
    3: "Язва",
    4: "Плодовое тело",
}
