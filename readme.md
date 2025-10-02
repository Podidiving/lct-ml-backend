# LCT ML BACKEND

### Сборка и запуск

в `app/weights` поместить `.pt` файлы (указано в `app/constants.py`)

Ссылка на скачивание: TBD (мы не выкладываем веса моделей до конца хакатона. Веса могут быть предоставлены организаторам по запросу через телеграмм, указанный в презентации)

#### CPU

```bash
docker build -t yolov11-cls:cpu -f Dockerfile-cpu .
docker run --rm -p 8123:8123 yolov11-cls:cpu
```


#### GPU

```bash
docker build -t yolov11-cls:gpu -f Dockerfile-gpu .
docker run --rm -p 8123:8123 --gpus all yolov11-cls:gpu
```


### Тестирование

`python send_request.py --image assets/example_tree.jpg`
