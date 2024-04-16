# diabetic_retinopathy_classification
Kaggle соревнование: [Diabetic Retinopathy Classification F1 Score #4](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/overview)
Результат: 4-е место (Ruslan Salikhzianov), Macro F1 score = 0.63060

## Introduction
В проекте представлена полная работа по подготовке модели для определения заболевания и степени тяжести [Диабетиической ретинопатиии](https://ru.wikipedia.org/wiki/%D0%94%D0%B8%D0%B0%D0%B1%D0%B5%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D1%80%D0%B5%D1%82%D0%B8%D0%BD%D0%BE%D0%BF%D0%B0%D1%82%D0%B8%D1%8F) на основе снимков глаз. Диабетиическая ретинопатиия — поражение сетчатки глаза диабетического происхождения. Является одним из наиболее тяжёлых осложнений сахарного диабета; проявляется в виде диабетической микроангиопатии, поражающей сосуды сетчатой оболочки глазного яблока, наблюдаемой у 90% пациентов при сахарном диабете.
![Пример болезни](https://github.com/salih-ds/diabetic_retinopathy_classification/style/example.png)

Работа проведена на основе соревнования kaggle: [Diabetic Retinopathy Classification F1 Score #4](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/overview). Целью соревнования является обучение модели для предсказания заболевания и степени тяжести Диабетиической ретинопатиии. В качестве данных представлены снимки глаз и метка степени заболевания:
- 0 - No DR
- 1 - Mild
- 2 - Moderate
- 3 - Severe
- 4 - Proliferative DR

Данные по классам не сбалансированы - половину примеров представляет класс 0. В качестве метрики используется "Macro F1 score", которая подходит для оценки успешности предсказания всех классов, а не наиболее представленных в примерах ввиду несбалансированности.

Результатом проекта является модель с Macro F1 score = 0.63060 (#4 в [leaderboard](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/leaderboard)).

## Data preparation
Представленные [в соревновании данные](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/data) несбалансированны. Для решения проблемы дополняю данные с аналогичного [соревнования 2015 года](https://www.kaggle.com/competitions/diabetic-retinopathy-detection), использую [пользовательский датасет](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) с уменьшенными изображениями.

В ходе подготовки данных получим сбалансированный тренировочный датасет с ~1000 изображений для каждого класса, при этом для валидационного датасета сохраняем несбалансированность для лучшей оценки.

Для выполнения необходимо запустить команду в консоли:
```
python data.py
```
Важно! Для исполнения кода необходимо загрузить архивы с данными [соревнования](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/data) с именем файла "diabetic-retinopathy-classification-f1-score-4.zip" и [дополнительные данные](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) с именем файла "train_image_resized.zip" и сохранить в рабочий каталог (Если имена архивов отличаются, то нужно переименовать, как указано выше для корректной работы команды)

Опциональные параметры:
- -m - абсолютный путь к архиву с данными соревнования (по умолчанию: "{dir}/diabetic-retinopathy-classification-f1-score-4.zip")
- -a - абсолютный путь к архиву с дополнительными данными (по умолчанию: "{dir}/train_image_resized.zip")
- -d - название директории, которая будет создана в рабочем каталоге и куда будут размещены данные из архивов (по умолчанию: "data")

Шаблон для исполнения команды с параметрами:
```
python data.py -m 'zip_main' -a 'zip_2015' -d 'directory'
```

## Model training

В ходе выбора core-архитектуры было принято решение об использовании efficientnet_v2_m, ввиду более быстрого схождения и лучшего результата основной метрики F1 на ограниченном количестве эпох.


| Модель            | Вес (МБ) | Benchmark imagenet | F1 val   |
|-------------------|----------|--------------------|----------|
| efficientnet_v2_m | 208      | 86.1               | 0.436873 |
| regnet_x_16gf     | 207      | 82                 | 0.397997 |
| swin_v2_s         | 190      | 83                 | 0.079304 |



Текущую задачу классификации болезни можно решать как через классификацию, так и через регрессионную модель. Это связано с тем, что метки - это числа, которые можно отсортировать в порядке возрастания, где каждая последующая метка соответствует все более тяжелой степени заболевания (0 - отсутствие заболевания, 4 - самая тяжелая степень заболевания). Исходя из опыта предыдущих аналогичных соревнований, где в лидерах чаще были регрессионные модели, было принято решение обучать именно регрессионную модель.

Устройство: NVIDIA Tesla V100 / 1 эпоха  ~= 15 мин
Модель: efficientnet_v2_m + pretrained imagenet weights, последний слой заменен на линейный с одним нейроном на выходе
Параметры обучения:
- image: 380x380 + augmentation + background cropping
- batch size: 18
- epoch: 30 (~ 7.5 часов)
- loss: SmoothL1Loss
- optimizer: AdamW, lr=0.001
- scheduler: multiply lr by 0.1 every 10 epoch

Команда для запуска обучения модели:
```
python train.py
```
Важно! Для запуска обучения сначала необходимо успешно выполнить data.py (см. выше). Данные для обучения должны лежать в директории "data", иначе определить вручную с помощью параметра -d (см. ниже).

Опциональные параметры:
- -b - размер батча (по умолчанию: 18)
- -e - количество эпох обучения (по умолчанию: 30)
- -d - название директории, в которой распакованы данные для обучения (по умолчанию: "data")
- -s - название директории, в которой будут сохраняться лучшие и последние веса модели (по умолчанию "weights")
- -n - название модели, используемое при сохранении весов (по умолчанию: "efficientnet_v2_m__size_380__crop__aug__adamw__lr_0001_10_01")

Шаблон для исполнения команды с параметрами:
```
python train.py -b 18 -e 30 -d 'data' -s 'save' -n 'name'
```

## Predict

Чтобы сделать предсказание, необходимо подготовить данные на вход модели. Для этого положите в директорию results/images изображения, для которых хотите выполнить предсказание. Результатом предсказания является файл results/result.csv, в котором содержатся Названия изображений и Предсказанные классы.

Также для выполнения предсказания необходимо обучить модель train.py (см. выше) или загрузить готовые [обученные веса для модели](https://drive.google.com/file/d/14m1W43eXiWA7-f96MoIQSs3kOlo-xhUA/view?usp=sharing) и положить в директорию "weights".

Для выполнения предсказания используйте следующую команду:
```
python predict.py
```

Опциональные параметры:
- -m - абсолютный путь к весам модели (по умолчанию: "{dir}/weights/efficientnet_v2_m__size_380__crop__aug__adamw__lr_0001_10_01__best.pt")

Шаблон для исполнения команды с параметрами:
```
python predict.py -m 'model_path'
```