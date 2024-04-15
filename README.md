# diabetic_retinopathy_classification
Kaggle competition: [Diabetic Retinopathy Classification F1 Score #4](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/overview)

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

Результатом проекта является модель с Macro F1 score = 0.63060 (#4 in the [leaderboard](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/leaderboard)).

## Data preparation
Представленные [в соревновании данные](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/data) несбалансированны. Для решения проблемы дополняю данные с аналогичного [соревнования 2015 года](https://www.kaggle.com/competitions/diabetic-retinopathy-detection), использую [пользовательский датасет](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) с уменьшенными изображениями.

В ходе подготовки данных получим сбалансированный тренировочный датасет с ~1000 изображений для каждого класса, при этом для валидационного датасета сохраняем несбалансированность для лучшей оценки.

Для выполнения необходимо запустить команду в консоли:
```
python data.py
```
Важно! Для исполнения кода необходимо загрузить архивы с данными [соревнования](https://www.kaggle.com/competitions/diabetic-retinopathy-classification-f1-score-4/data) с именем файла "diabetic-retinopathy-classification-f1-score-4.zip" и [дополнительные данные](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized) с именем файла "train_image_resized.zip" и сохранить в рабочий каталог (Если имена архивов отличаются, то нужно переименовать, как указано выше для корректной работы команды)

Опциональные параметры:
- -m - указать абсолютный путь к архиву с данными соревнования (по умолчанию: "{dir}/diabetic-retinopathy-classification-f1-score-4.zip")
- -a - указать абсолютный путь к архиву с дополнительными данными (по умолчанию: "{dir}/train_image_resized.zip")
- -d - название директории, которая будет создана в рабочем каталоге и куда будут размещены данные из архивов (по умолчанию: "data")

Шаблон для исполнения команды с параметрами:
```
python data.py -m 'value' -a 'value' -d 'value'
```