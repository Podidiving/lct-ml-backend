# LCT ML BACKEND

### Сборка и запуск

в `app/weights` поместить `.pt` файл (как он называется указано в `app/constants.py`)

#### CPU

```bash
docker build -t yolov11-cls:cpu -f Dockerfile-cpu .
docker run --rm -p 8123:8123 yolov11-cls:cpu
```


#### GPU

(не тестилось, мб надо будет поправить библиотекик)

```bash
docker build -t yolov11-cls:gpu -f Dockerfile-gpu .
docker run --rm -p 8123:8123 --gpus all yolov11-cls:gpu
```